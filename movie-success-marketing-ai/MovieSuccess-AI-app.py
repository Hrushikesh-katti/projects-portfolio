import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from openai import OpenAI
import gradio as gr

# Step 1: Load the dataset
print("Loading dataset...")
df = pd.read_csv("movie_metadata.csv")

# Step 2: Data Cleaning
print("Cleaning data...")

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"After removing duplicates, dataset has {df.shape[0]} rows")

# Handle missing values
numerical_cols = ['budget', 'gross', 'num_voted_users', 'num_critic_for_reviews', 
                  'num_user_for_reviews', 'duration', 'movie_facebook_likes', 'title_year']
df['title_year'] = pd.to_numeric(df['title_year'], errors='coerce')
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = ['content_rating', 'language', 'country', 'genres']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df['plot_keywords'] = df['plot_keywords'].fillna('unknown')

# Cap outliers
skewed_cols = ['budget', 'gross', 'num_voted_users', 'num_critic_for_reviews', 
               'num_user_for_reviews', 'movie_facebook_likes']
for col in skewed_cols:
    df[col] = np.log1p(df[col])
    cap_value = np.percentile(df[col], 99)
    df[col] = df[col].clip(upper=cap_value)

df['duration'] = df['duration'].clip(lower=60, upper=180)

print("\n=== After Data Cleaning ===")
print(df.info())

# Step 3: Feature Engineering
print("Performing feature engineering...")

# Categorize target variable (Flop: <5.5, Average: 5.5-7, Hit: >7)
def categorize_imdb_score(score):
    if not isinstance(score, (int, float)):
        return 'Unknown'
    if score < 5.5:
        return 'Flop'
    elif 5.5 <= score <= 7:
        return 'Average'
    elif score > 7:
        return 'Hit'
    return 'Unknown'

df['Classify'] = df['imdb_score'].apply(categorize_imdb_score)
mode_class = df[df['Classify'] != 'Unknown']['Classify'].mode()[0]
df['Classify'] = df['Classify'].replace('Unknown', mode_class)
df = df.drop('imdb_score', axis=1)

print("\n=== Class Distribution ===")
print(df['Classify'].value_counts())

# Drop irrelevant columns (keep plot_keywords for tagline generation)
irrelevant_cols = ['movie_title', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 
                   'movie_imdb_link', 'actor_1_facebook_likes', 'actor_2_facebook_likes',
                   'actor_3_facebook_likes', 'cast_total_facebook_likes', 'director_facebook_likes',
                   'aspect_ratio', 'facenumber_in_poster', 'color']
existing_irrelevant_cols = [col for col in irrelevant_cols if col in df.columns]
df = df.drop(columns=existing_irrelevant_cols)

# Create new features
df['budget_to_gross_ratio'] = df['budget'] / df['gross']
df['budget_to_gross_ratio'] = df['budget_to_gross_ratio'].replace([np.inf, -np.inf], np.nan)
df['budget_to_gross_ratio'] = df['budget_to_gross_ratio'].fillna(df['budget_to_gross_ratio'].median())

df['genre_count'] = df['genres'].str.split('|').apply(len)

print("\n=== After Feature Engineering ===")
print(df.info())

# Step 4: Preprocess the Data for Training
print("Preprocessing data for training...")

# Define features and target
X = df.drop(['Classify', 'plot_keywords', 'genres'], axis=1)  # Exclude plot_keywords and genres from features
y = df['Classify']

# Encode categorical features
le_content_rating = LabelEncoder()
valid_ratings = ['G', 'PG', 'PG-13', 'R']
df['content_rating'] = df['content_rating'].apply(lambda x: x if x in valid_ratings else 'Other')
X['content_rating'] = le_content_rating.fit_transform(df['content_rating'])

# One-hot encode language and country (without drop_first)
df['language'] = df['language'].apply(lambda x: 'English' if x.lower() == 'english' else 'Other')
df['country'] = df['country'].apply(lambda x: 'USA' if x.upper() == 'USA' else 'Other')
X = pd.get_dummies(X, columns=['language', 'country'], prefix=['lang', 'country'])

# Encode genres using MultiLabelBinarizer
mlb_genres = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(
    mlb_genres.fit_transform(df['genres'].str.split('|')),
    columns=mlb_genres.classes_,
    index=df.index
)
X = pd.concat([X, genres_encoded], axis=1)

# Scale numerical features
numerical_cols = ['num_critic_for_reviews', 'duration', 'gross', 'budget', 'num_voted_users', 
                  'num_user_for_reviews', 'movie_facebook_likes', 'title_year', 
                  'budget_to_gross_ratio', 'genre_count']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Ensure all columns are numeric
X = X.astype(float)

# Step 5: Train the RandomForest Model
print("Training RandomForest model...")
rf_model = RandomForestClassifier(
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)
rf_model.fit(X, y)

# Step 6: Get Default Values for Optional Inputs
default_values = { 
    'num_critic_for_reviews': df['num_critic_for_reviews'].median(),
    'num_voted_users': df['num_voted_users'].median(),
    'num_user_for_reviews': df['num_user_for_reviews'].median(),
    'movie_facebook_likes': df['movie_facebook_likes'].median()
}

# Step 7: Set up DeepSeek API via OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="[Replace with your API key here]" #you can get it from https://openrouter.ai
)

# Step 8: Define the Prediction and Marketing Generation Function
def predict_and_generate_marketing(genres, plot_keywords, budget, gross, duration, title_year, content_rating, language, country,
                                   num_critic_for_reviews=None, num_voted_users=None, num_user_for_reviews=None, movie_facebook_likes=None):
    # Use default values if optional inputs are not provided
    num_critic_for_reviews = num_critic_for_reviews if num_critic_for_reviews is not None else default_values['num_critic_for_reviews']
    num_voted_users = num_voted_users if num_voted_users is not None else default_values['num_voted_users']
    num_user_for_reviews = num_user_for_reviews if num_user_for_reviews is not None else default_values['num_user_for_reviews']
    movie_facebook_likes = movie_facebook_likes if movie_facebook_likes is not None else default_values['movie_facebook_likes']

    # Validate inputs
    if not all([genres, plot_keywords, budget, gross, duration, title_year, content_rating, language, country]):
        return "Error: All required fields must be filled.", "", ""

    # Validate content_rating
    valid_ratings = ['G', 'PG', 'PG-13', 'R', 'Other']
    if content_rating not in valid_ratings:
        return f"Error: Invalid content rating '{content_rating}'. Must be one of {valid_ratings}.", "", ""

    # Create user DataFrame
    user_data = {
        'num_critic_for_reviews': num_critic_for_reviews,
        'duration': duration,
        'gross': gross,
        'budget': budget,
        'num_voted_users': num_voted_users,
        'num_user_for_reviews': num_user_for_reviews,
        'movie_facebook_likes': movie_facebook_likes,
        'title_year': title_year,
        'content_rating': content_rating,
        'language': language,
        'country': country,
        'genres': genres
    }
    user_df = pd.DataFrame([user_data])

    # Preprocess user inputs
    user_df['budget_to_gross_ratio'] = user_df['budget'] / user_df['gross']
    user_df['budget_to_gross_ratio'] = user_df['budget_to_gross_ratio'].replace([np.inf, -np.inf], np.nan)
    user_df['budget_to_gross_ratio'] = user_df['budget_to_gross_ratio'].fillna(user_df['budget_to_gross_ratio'].median())

    user_df['genre_count'] = user_df['genres'].str.split('|').apply(len)

    # Apply log transformation and capping
    skewed_cols = ['budget', 'gross', 'num_voted_users', 'num_critic_for_reviews', 
                   'num_user_for_reviews', 'movie_facebook_likes']
    for col in skewed_cols:
        user_df[col] = np.log1p(user_df[col])
        cap_value = np.percentile(df[col], 99)
        user_df[col] = user_df[col].clip(upper=cap_value)

    user_df['duration'] = user_df['duration'].clip(lower=60, upper=180)

    # Encode categorical features
    user_df['content_rating'] = user_df['content_rating'].apply(lambda x: x if x in valid_ratings else 'Other')
    user_df['content_rating'] = le_content_rating.transform(user_df['content_rating'])

    user_df['language'] = user_df['language'].apply(lambda x: 'English' if x.lower() == 'english' else 'Other')
    user_df['country'] = user_df['country'].apply(lambda x: 'USA' if x.upper() == 'USA' else 'Other')
    user_df = pd.get_dummies(user_df, columns=['language', 'country'], prefix=['lang', 'country'])

    # Ensure all dummy columns exist
    for col in ['lang_English', 'lang_Other', 'country_USA', 'country_Other']:
        if col not in user_df.columns:
            user_df[col] = 0

    # Encode genres
    genres_list = user_df['genres'].str.split('|').iloc[0]
    genres_encoded = pd.DataFrame(
        mlb_genres.transform([genres_list]),
        columns=mlb_genres.classes_
    )
    user_df = pd.concat([user_df, genres_encoded], axis=1)

    # Scale numerical features
    user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])

    # Prepare features for prediction
    feature_names = X.columns.tolist()
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0
    X_user = user_df[feature_names]

    # Predict success
    predicted_success = rf_model.predict(X_user)[0]

    # Generate marketing taglines and strategies
    tagline_prompt = (
        f"Generate 5 short, catchy marketing taglines for a movie with the following details:\n"
        f"- Predicted Success: {predicted_success}\n"
        f"- Genres: {genres}\n"
        f"- Plot Keywords: {plot_keywords}\n"
        f"Provide the taglines as a numbered list:\n"
        f"1. First tagline\n"
        f"2. Second tagline\n"
        f"3. Third tagline\n"
        f"4. Fourth tagline\n"
        f"5. Fifth tagline\n"
    )
    tagline_response = client.chat.completions.create(
        model="deepseek/deepseek-chat:free",
        messages=[
            {"role": "system", "content": "You are a creative marketing assistant specializing in movie promotions."},
            {"role": "user", "content": tagline_prompt}
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=500
    )
    tagline_text = tagline_response.choices[0].message.content

    taglines = []
    lines = tagline_text.split('\n')
    for line in lines:
        if line.strip().startswith(tuple(f"{i}." for i in range(1, 6))):
            tagline = line.split('.', 1)[1].strip()
            if tagline and tagline != "First tagline" and tagline != "Second tagline" and tagline != "Third tagline" and tagline != "Fourth tagline" and tagline != "Fifth tagline":
                taglines.append(tagline)
    taglines_text = "Could not generate meaningful taglines." if not taglines else "\n".join(f"{i+1}. {tagline}" for i, tagline in enumerate(taglines[:5]))

    strategy_prompt = (
        f"Generate a marketing strategy for a movie with the following details:\n"
        f"- Predicted Success: {predicted_success}\n"
        f"- Genres: {genres}\n"
        f"- Plot Keywords: {plot_keywords}\n"
        f"Based on the predicted success:\n"
        f"- If a Hit: Focus on maintaining hype and maximizing reach.\n"
        f"- If Average: Suggest ways to boost its success and appeal.\n"
        f"- If a Flop: Propose a plan to recover investment and minimize losses.\n"
        f"Start your response with 'Marketing Strategy:' and provide a concise strategy in 2-3 sentences.\n"
    )
    strategy_response = client.chat.completions.create(
        model="deepseek/deepseek-chat:free",
        messages=[
            {"role": "system", "content": "You are a creative marketing assistant specializing in movie promotions."},
            {"role": "user", "content": strategy_prompt}
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=500
    )
    strategy_text = strategy_response.choices[0].message.content

    strategy_start = strategy_text.find("Marketing Strategy:")
    if strategy_start != -1:
        strategy_text = strategy_text[strategy_start+len("Marketing Strategy:"):].strip()
    else:
        strategy_text = strategy_text.strip()
    strategy_sentences = strategy_text.split('.')
    strategy_text = '. '.join([s.strip() for s in strategy_sentences[:3] if s.strip()]) + ('.' if strategy_text.strip() else '')
    if not strategy_text:
        strategy_text = "Could not generate a meaningful strategy."

    return predicted_success, taglines_text, strategy_text

# ... [keep all previous code until the Gradio interface] ...

# Step 9: Set up the Gradio Interface
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="orange",
        font=gr.themes.GoogleFont("Inter")
    ),
    title="Movie Success Predictor"
) as demo:
    
    gr.Markdown("""
    <div style='text-align: center'>
        <h1>🎬 Movie Success Predictor</h1>
        <p>Predict success category and generate marketing materials</p>
    </div>
    """)
    
    with gr.Row():
        # Input Column
        with gr.Column(scale=2):
            gr.Markdown("**Movie Information**")
            genres = gr.Textbox(label="Genres (separate with |)", value="Action|Adventure")
            plot_keywords = gr.Textbox(label="Plot Keywords (separate with |)", value="hero|battle|future")
            
            with gr.Row():
                budget = gr.Number(label="Budget ($)", value=50000000)
                gross = gr.Number(label="Gross Revenue ($)", value=100000000)
            
            with gr.Row():
                duration = gr.Number(label="Duration (mins)", value=120)
                title_year = gr.Number(label="Release Year", value=2024)
            
            content_rating = gr.Dropdown(
                label="Content Rating", 
                choices=['G', 'PG', 'PG-13', 'R', 'Other'],
                value='PG-13'
            )
            
            # Optional fields
            with gr.Accordion("Additional Options", open=False):
                language = gr.Textbox(label="Language", value="English")
                country = gr.Textbox(label="Country", value="USA")
                num_critic_for_reviews = gr.Number(label="Critic Reviews", value=200)
                num_voted_users = gr.Number(label="IMDb Votes", value=5000)
                movie_facebook_likes = gr.Number(label="Facebook Likes", value=10000)
        
        # Results Column
        with gr.Column(scale=1):
            gr.Markdown("**Results**")
            prediction = gr.Textbox(label="Predicted Success", interactive=False)
            
            gr.Markdown("**Marketing Content**")
            taglines = gr.Textbox(label="Taglines", lines=3, interactive=False)
            strategy = gr.Textbox(label="Strategy", lines=3, interactive=False)
    
    submit_btn = gr.Button("Analyze Movie", variant="primary")
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("*ML Model: RandomForest | LLM: DeepSeek-v3*")

    submit_btn.click(
        fn=predict_and_generate_marketing,
        inputs=[
            genres, plot_keywords, budget, gross, duration, title_year,
            content_rating, language, country, num_critic_for_reviews,
            num_voted_users, movie_facebook_likes
        ],
        outputs=[prediction, taglines, strategy]
    )

demo.launch()