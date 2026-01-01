from github import Auth, Github
from llama_index.core.workflow import Context
# In LlamaIndex, we can use classes like FunctionAgent or ReActAgent or CodeAgent to create various types of agents
# from llama_index.core.agent.workflow import ReActAgent # Standalone agent
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow, ToolCall, ToolCallResult, AgentOutput
from llama_index.core.tools import FunctionTool
from llama_index.core.prompts import RichPromptTemplate
from typing import Any
import asyncio
from llama_index.llms.openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()

# -----------------------------
# Setup LLM
# -----------------------------

llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

# -----------------------------
# Setup GitHub client
# -----------------------------
#Locally → uses your PAT, CI → uses GitHub’s auto token, If nothing is set → falls back to anonymous
github_client = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else Github()

# REPOSITORY and PR number provided by env (local or CI)
repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

# Load the repo object
repo = github_client.get_repo(repository)

# Construct repo URL for test
repo_url = f"https://github.com/{repository}"

# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------

def get_pr_details(pr_number: int):
    """
    Retrieve basic information about a pull request.

    Returns a dictionary containing:
        - author: PR author username
        - title: PR title
        - body: PR description text
        - diff_url: URL to the PR diff
        - state: PR state (open/closed)
        - commit_SHAs: list of all commit SHAs in the PR

    Note:
        The PR 'body' is not reliable for detecting changed files.
        To get actual file changes, call pr_commit_details() for each SHA
        in commit_SHAs.
    """

    try :
        pull_request = repo.get_pull(pr_number)
        return {
            "author": pull_request.user.login,
            "title": pull_request.title,
            "body": pull_request.body,
            "diff_url": pull_request.diff_url,
            "state": pull_request.state,
            # "head_sha": pull_request.head.sha,
            "commit_SHAs": [c.sha for c in pull_request.get_commits()],
        }
    except Exception:
        return {
            "author": "unknown",
            "title": "unknown",
            "body": "",
            "diff_url": "unknown",
            "state": "unknown",
            "commit_SHAs": [],
        }


def get_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """
    Retrieve the list of changed files for a specific commit SHA.

    Each returned entry includes:
        - filename: full path of the changed file
        - status: change type (added / modified / removed)
        - additions: number of added lines
        - deletions: number of removed lines
        - changes: total line changes
        - patch: unified diff patch (may be None)
        - ref: the commit SHA this file change belongs to

    Note:
        The 'ref' field is included so the agent can fetch the correct
        version of each file using get_file_content().

    Returns:
        A list of dictionaries describing all files changed in the commit.
    """

    commit = repo.get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []

    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
            "ref": head_sha,
        })

    return changed_files


def get_file_content(file_path: str, ref: str) -> str | None:
    """
    Retrieve the contents of a file at a specific commit SHA.

    Args:
        file_path: Full file path including the file name, exactly as returned
                   by get_commit_details() (e.g., "src/utils/helpers.py").
        ref: Commit SHA to fetch the file from.

    Returns:
        The decoded file content as a string, or None if the file does not
        exist at that ref or is not a regular file.
    """
    try:
        file_content = repo.get_contents(file_path, ref=ref)

        # GitHub returns a list if the path is a directory
        if isinstance(file_content, list):
            return None

        if file_content.type != "file":
            return None

        return file_content.decoded_content.decode("utf-8")

    except Exception:
        return None

async def add_gathered_context_to_state(ctx: Context, gathered_context: str) -> None:
    """
    Store the gathered pull‑request context in the shared agent state.

    The `gathered_context` argument should be a clear natural‑language summary
    containing all relevant PR information (metadata, changed files, commit notes,
    and any extracted content). This text will be used by other agents to reason
    about the PR and generate their analysis or review.
    """
    current_state = await ctx.store.get("state")
    current_state["gathered_context"] = gathered_context
    await ctx.store.set("state", current_state)

async def add_draft_comment_to_state(ctx: Context, draft_comment: str) -> None:
    """
    Store the draft PR comment in the shared agent state.

    The draft_comment string should contain the full comment generated by
    the CommentorAgent before final submission. Other agents may read or
    modify this value before the workflow completes.
    """
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)

async def add_final_review_to_state(ctx: Context, final_review_comment: str) -> None:
    """
    Store the final review comment in the agent's shared state.

    This function updates the `final_review_comment` field inside the
    persistent state dictionary managed by the agent framework. It is
    asynchronous because state access (`ctx.store.get` / `ctx.store.set`)
    uses async I/O under the hood.

    Args:
        ctx (Context): The agent execution context, providing access to state.
        final_review_comment (str): The finalized review text to store.
    """
    current_state = await ctx.store.get("state")  # type: ignore
    current_state["final_review_comment"] = final_review_comment
    await ctx.store.set("state", current_state)  # type: ignore

def post_final_review_to_github(pr_number: int, final_review_comment: str):
    """
    Post the final review comment to a GitHub pull request.

    This function retrieves the pull request using the global `repo`
    object and submits a review comment using the GitHub API.

    Args:
        pr_number (int): The pull request number.
        final_review_comment (str): The final review comment to post.

    Raises:
        Exception: If the GitHub API call fails.
    """
    try:
        pr = repo.get_pull(pr_number)
        pr.create_review(body=final_review_comment, event="COMMENT")
    except Exception as e:
        raise e

# -----------------------------
# Convert Functions to Tools
# -----------------------------
get_pr_details_tool = FunctionTool.from_defaults(
    fn=get_pr_details,
    name="get_pr_details",
    description="Retrieve details of a GitHub pull request by number."
)

get_commit_details_tool = FunctionTool.from_defaults(
    fn=get_commit_details,
    name="get_commit_details",
    description="Retrieve commit details (files, stats, patch) by commit SHA."
)

get_file_contents_tool = FunctionTool.from_defaults(
    fn=get_file_content,
    name="get_file_content",
    description="Retrieve the contents of a file from the repository."
)

add_gathered_context_to_state_tool = FunctionTool.from_defaults(
    fn=add_gathered_context_to_state,
    name="add_gathered_context_to_state",
    # description="Save gathered context into the shared agents state."
    description="Store a natural‑language summary of all gathered PR information in shared state so other agents can use it.",
)

add_draft_comment_to_state_tool = FunctionTool.from_defaults(
    fn=add_draft_comment_to_state,
    name="add_draft_comment_to_state",
    description="Save draft comment into the shared agents state."
)

add_final_review_to_state_tool = FunctionTool.from_defaults(
    fn=add_final_review_to_state,
    name="add_final_review_to_state",
    description="Store the final PR review comment in shared state so the workflow can track and finalize the review."
)

post_final_review_to_github_tool = FunctionTool.from_defaults(
    fn=post_final_review_to_github,
    name="post_final_review_to_github",
    description="Post the finalized PR review comment to GitHub by creating a review comment on the specified pull request."
)

# ------------------------------------------------------------
# Agents
# ------------------------------------------------------------

context_prompt = """
You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed file names (including their paths and commit SHAs); \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
"""

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers PR details, commit info, and file contents.",
    tools=[get_pr_details_tool,
           get_commit_details_tool,
           get_file_contents_tool,
           add_gathered_context_to_state_tool],
    can_handoff_to=["CommentorAgent"],
    system_prompt=context_prompt,
)

commentor_prompt = """
You are the commentor agent that writes review comments for pull requests as a human reviewer would. 
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - If you need any additional details, you must hand off to the ContextAgent. Do NOT ask the user!
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? Use the diff to determine this.
    - Are new endpoints documented? Use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - You should directly address the author. So your comments should sound like:
   "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
 - **When handing off, you MUST call the `handoff` tool with:**
   **{"to_agent": "ReviewAndPostingAgent", "reason": "Draft review completed"}**
 - **Do NOT output a final response. Always call the handoff tool instead.**
"""

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Drafts a detailed pull request review using context from the ContextAgent, requesting additional information when needed, saving the draft review to shared state, and signaling readiness for the ReviewAndPostingAgent to finalize it.",
    tools=[add_draft_comment_to_state_tool],
    system_prompt=commentor_prompt,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_prompt = """
You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.
"""

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Oversee the quality of the draft PR review, coordinate rewrites with the CommentorAgent when needed, finalize the review, update shared state, and publish the approved comment to GitHub.",
    tools=[add_final_review_to_state_tool, post_final_review_to_github_tool],
    system_prompt=review_and_posting_prompt,
    can_handoff_to=["CommentorAgent"],
)

# ------------------------------------------------------------
# Workflow
# ------------------------------------------------------------

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review_comment": "",
    },
)

# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------
async def main():
    query = f"Write a review for PR: {pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

if __name__ == "__main__":
    asyncio.run(main())
    github_client.close()