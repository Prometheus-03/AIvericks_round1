# Load your OpenAI API key
OPENAI_API_KEY = "APIKEY"
import openai
import pandas as pd
import time

# Load the CSV file
df = pd.read_csv("data/All_Diets.csv")

# Ensure the correct column name is used for dish names
DISH_NAME_COLUMN = "Recipe_name"  # Updated column name
RECIPE_COLUMN = "AI Recipe"

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

def generate_recipe(dish_name):
    """Generates a step-by-step recipe using OpenAI API."""
    prompt = f"""
    Generate a step-by-step recipe for {dish_name}. Include:
    - Ingredients
    - Step-by-step cooking instructions
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional chef providing easy-to-follow recipes using the given ingredients."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        recipe = response.choices[0].message.content.strip()
        print(f"Recipe generated for: {dish_name}\n{recipe}\n")
        return recipe
    except Exception as e:
        print(f"Error generating recipe for {dish_name}: {e}")
        return "Error generating recipe."

# Add the AI-generated recipes
df[RECIPE_COLUMN] = df[DISH_NAME_COLUMN].apply(generate_recipe)

# Save the updated CSV file
df.to_csv("data/All_Diets_with_Recipes.csv", index=False)

print("Updated CSV file saved as 'All_Diets_with_Recipes.csv'")
