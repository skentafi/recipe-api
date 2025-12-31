from rest_framework import serializers

from .models import Recipe

# Stage 5 test change

class RecipeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recipe
        fields = ["id", "title", "description", "time_minutes", "price"]
