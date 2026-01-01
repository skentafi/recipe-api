from rest_framework import serializers

from .models import Recipe

class RecipeSerializer(serializers.ModelSerializer):
    # Serializer for converting Recipe model instances into JSON-friendly data
    class Meta:
        model = Recipe
        fields = ["id", "title", "description", "time_minutes", "price"]
