function titleCase(text){
    return text.charAt(0).toUpperCase() + text.substr(1).toLowerCase();
}

function generateFoodSuggestions() {
    var age = document.getElementById("age").value;
    var gender = document.getElementById("gender").value;
    var height = document.getElementById("height").value;
    var weight = document.getElementById("weight").value;
    var activity = document.getElementById("activity").value;

    if (!age || !gender || !height || !weight || !activity) {
        alert("Please fill in all fields.");
        return;
    }

    var foodRecipes = [
        {
            name: "Chicken Breast with Roasted Broccoli",
            ingredients: [
                "1 boneless, skinless chicken breast",
                "1 cup broccoli florets",
                "1 tbsp olive oil",
                "Salt and pepper to taste"
            ],
            steps: [
                "Preheat oven to 400°F (200°C).",
                "Toss broccoli with olive oil, salt, and pepper.",
                "Place chicken and broccoli on a baking sheet.",
                "Bake for 20-25 minutes, or until chicken is cooked through."
            ],
            nutrition: {
                calories: "Approx. 350 kcal",
                protein: "40g",
                fat: "15g",
                carbs: "10g"
            }
        },
        {
            name: "Salmon with Quinoa",
            ingredients: [
                "4 oz salmon fillet",
                "1/2 cup cooked quinoa",
                "1 tbsp lemon juice",
                "1 tsp dill",
                "Salt and pepper to taste"
            ],
            steps: [
                "Season salmon with lemon juice, dill, salt, and pepper.",
                "Pan-fry or bake salmon until cooked through.",
                "Serve with cooked quinoa."
            ],
            nutrition: {
                calories: "Approx. 400 kcal",
                protein: "30g",
                fat: "20g",
                carbs: "25g"
            }
        },
        {
            name: "Lentil Soup",
            ingredients: [
                "1 cup red lentils",
                "4 cups vegetable broth",
                "1 onion, chopped",
                "2 carrots, chopped",
                "2 celery stalks, chopped",
                "1 tsp cumin",
                "Salt and pepper to taste"
            ],
            steps: [
                "Rinse lentils.",
                "Sauté onion, carrots, and celery in a pot.",
                "Add lentils, vegetable broth, cumin, salt, and pepper.",
                "Bring to a boil, then simmer for 20-25 minutes, or until lentils are tender."
            ],
            nutrition: {
                calories: "Approx. 250 kcal",
                protein: "15g",
                fat: "5g",
                carbs: "40g"
            }
        },
        {
            name: "Oatmeal with Banana and Almonds",
            ingredients: [
                "1/2 cup rolled oats",
                "1 cup milk or water",
                "1/2 banana, sliced",
                "1/4 cup almonds, slivered",
                "1 tsp honey (optional)"
            ],
            steps: [
                "Cook oats with milk or water according to package directions.",
                "Top with sliced banana and almonds.",
                "Drizzle with honey if desired."
            ],
            nutrition: {
                calories: "Approx. 300 kcal",
                protein: "10g",
                fat: "12g",
                carbs: "40g"
            }
        },
        {
            name: "Egg and Avocado Toast",
            ingredients: [
                "2 slices whole-wheat toast",
                "1 avocado, mashed",
                "2 eggs",
                "Salt and pepper to taste",
                "Red pepper flakes (optional)"
            ],
            steps: [
                "Toast bread.",
                "Mash avocado and spread on toast.",
                "Cook eggs to your liking (fried, scrambled, poached).",
                "Place eggs on avocado toast, season with salt, pepper, and red pepper flakes if desired."
            ],
            nutrition: {
                calories: "Approx. 380 kcal",
                protein: "18g",
                fat: "25g",
                carbs: "30g"
            }
        }
    ];

    var numberOfSuggestions = 3;
    var suggestedFoods = [];

    for (let i = 0; i < numberOfSuggestions; i++) {
        var randomIndex = Math.floor(Math.random() * foodRecipes.length);
        suggestedFoods.push(foodRecipes[randomIndex]);
        foodRecipes.splice(randomIndex, 1);
    }


    var foodList = document.getElementById("foodList");
    foodList.innerHTML = "";

    suggestedFoods.forEach(function(food) {
        var listItem = document.createElement("li");

        var foodName = document.createElement("h3");
        foodName.textContent = food.name;
        listItem.appendChild(foodName);

        var ingredientsList = document.createElement("ul");
        var ingredientsHeader = document.createElement("h4");
        ingredientsHeader.textContent = "Ingredients:";
        listItem.appendChild(ingredientsHeader);
        food.ingredients.forEach(function(ingredient) {
            var ingredientItem = document.createElement("li");
            ingredientItem.textContent = ingredient;
            ingredientsList.appendChild(ingredientItem);
        });
        listItem.appendChild(ingredientsList);


        var stepsList = document.createElement("ol");
        var stepsHeader = document.createElement("h4");
        stepsHeader.textContent = "Steps:";
        listItem.appendChild(stepsHeader);
        food.steps.forEach(function(step) {
            var stepItem = document.createElement("li");
            stepItem.textContent = step;
            stepsList.appendChild(stepItem);
        });
        listItem.appendChild(stepsList);

        var nutritionDiv = document.createElement("div");
        nutritionDiv.classList.add("nutrition-info");
        
        var nutritionHeader = document.createElement("h4");
        nutritionHeader.textContent = "Nutritional Information (Per Serving):";
        nutritionDiv.appendChild(nutritionHeader);

        var nutritionList = document.createElement("ul");
        for (const nutrient in food.nutrition) {
            var nutrientItem = document.createElement("li");
            nutrientItem.textContent = `${titleCase(nutrient)}: ${food.nutrition[nutrient]}`;
            nutritionList.appendChild(nutrientItem);
        }
        nutritionDiv.appendChild(nutritionList);
        listItem.appendChild(nutritionDiv);
        foodList.appendChild(listItem);
    });
}