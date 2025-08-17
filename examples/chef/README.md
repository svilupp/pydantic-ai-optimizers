# Chef Agent Example

A comprehensive food recommendation agent built with modern PydanticAI patterns, featuring allergen safety, dietary restrictions, and recipe search capabilities.

## 🏗️ Architecture

### Modern PydanticAI Structure
- **agent.py**: Clean agent definition with `@agent.tool` decorators and union output types
- **dataset.py**: 10 diverse evaluation cases using Pydantic Evals framework
- **evaluators.py**: Field-by-field evaluators using PydanticAI Helpers directly
- **optimize.py**: Simplified optimization logic with proper reflection handling

### Key Features
- ✅ `ResponseToUser | ServiceRejection` union output types
- ✅ `ChefDeps` dataclass for dependency injection
- ✅ `ph.History(result)` for proper tool call extraction
- ✅ Direct PydanticAI Helpers evaluators (no custom wrappers)
- ✅ Fuzzy string matching and numeric tolerances
- ✅ MaxCount/MinCount for tool call bounds

## 📋 Evaluation Cases

The dataset includes 10 comprehensive test cases covering:

1. **Simple allergen avoidance** - Gluten allergy with pasta
2. **Multi-filter constraints** - Vegan Italian under budget with time limits
3. **Allergen + ingredients** - Tree nut allergy with specific ingredient requests
4. **Diet + calorie constraints** - Carnivore diet with calorie requirements
5. **Diet + allergen combo** - Pescatarian with shellfish allergy
6. **Complex multi-allergen** - Dairy + egg allergies with cuisine/difficulty
7. **Rejection scenarios** - Impossible/contradictory requests
8. **Multiple allergens + cuisine** - Soy + sesame allergies with Asian food
9. **Time + difficulty** - Quick cooking with specific difficulty level
10. **Comprehensive constraints** - All filters combined in one request

## 🚀 Usage

### 1. Test Individual Components

```bash
# Test dataset creation
uv run python dataset.py

# Test evaluators
uv run python evaluators.py
```

### 2. Run Evaluation Only (Recommended First Step)

Test the evaluation system without full optimization:

```bash
uv run python optimize.py --eval-only
```

This will:
- Load the dataset from `evals/chef_cases.yaml`
- Create the agent with the seed prompt
- Run evaluation on the first 2 cases
- Show detailed evaluation results

### 3. Run Full Optimization

Once evaluation works, run the complete optimization:

```bash
uv run python optimize.py
```

This will:
- Run the full evaluation dataset
- Use genetic optimization to improve prompts
- Save the best prompt to the prompt pool

## 🧪 Evaluation Framework

### PydanticAI Helpers Evaluators

The system uses direct PydanticAI Helpers evaluators:

```python
# Tool call validation
MaxCount(output_path="num_tool_calls", count=5)
MinCount(output_path="num_tool_calls", count=1)

# Field-by-field matching with fuzzy support
ScalarEquals(
    output_path="search_recipes_calls.0.query",
    expected_path="search_recipes_calls.0.query",
    fuzzy_enabled=True,
    fuzzy_threshold=0.8
)

# List comparisons with recall metrics
ListRecall(
    output_path="search_recipes_calls.0.ingredients_to_avoid",
    expected_path="search_recipes_calls.0.ingredients_to_avoid",
    fuzzy_enabled=True
)
```

### Tool Call Extraction

Uses proper PydanticAI Helpers for tool inspection:

```python
import pydantic_ai_helpers as ph

hist = ph.History(result)
allergen_calls = hist.tools.calls(name="search_allergens").all()
all_calls = hist.tools.calls().all()  # For ordering checks
```

## 📊 Expected Evaluation Metrics

The evaluators check:

- **Allergen call ordering** - Safety-critical allergen searches before recipes
- **Service decisions** - Proper recommend/reject logic
- **Tool call efficiency** - Bounded tool usage (1-5 calls)
- **Parameter accuracy** - Field-by-field search parameter matching
- **Ingredient safety** - Proper allergen avoidance lists
- **Constraint compliance** - Diet, cuisine, time, cost, calorie filters

## 🔧 Configuration

### Environment Setup

1. Set API keys in `.env` file:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

2. Configure models in optimizer config (if needed)

### Data Files

- `data/recipes.json` - 50+ recipes with nutritional info
- `data/allergens.json` - Allergen-to-ingredient mappings
- `prompts/chef_seed.txt` - Initial system prompt
- `evals/chef_cases.yaml` - Serialized evaluation dataset

## 🎯 Success Criteria

A successful evaluation run should show:
- High scores on allergen safety evaluations
- Good parameter matching with fuzzy tolerance
- Appropriate tool call counts (1-5 per case)
- Correct service decisions (recommend vs reject)
- Proper ingredient avoidance for allergies

The system prioritizes safety over perfect parameter matching, with allergen handling being the most critical evaluation criterion.