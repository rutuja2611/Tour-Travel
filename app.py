from flask import Flask, render_template, redirect, request, session, url_for
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from sqlalchemy import DateTime
from sqlalchemy import Numeric
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func, desc  # Import func and desc from SQLAlchemy
from sklearn.preprocessing import MinMaxScaler
from flask import Flask
from flask_login import LoginManager, login_required, UserMixin, current_user
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from flask import session


app = Flask(__name__)
login_manager = LoginManager()
login_manager.init_app(app)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # Use SQLite for simplicity
db = SQLAlchemy(app)


# user database created while registration-->WORKS
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    age = db.Column(db.Integer, nullable=True)  
    gender = db.Column(db.String(10), nullable=True) 
    country = db.Column(db.String(50), nullable=False)
    city = db.Column(db.String(50), nullable=False)
    area_code = db.Column(db.String(10), nullable=False)
    password = db.Column(db.String(60), nullable=False)


# hotel review database-->WORKS
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # place_name = db.Column(db.String(100), nullable=False)
    hotel_name = db.Column(db.String(100), nullable=False)
    rating = db.Column(Numeric(precision=3, scale=1), nullable=False)
    review_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('reviews', lazy=True))

# in order to save the preferences of the user.
class SearchQuery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    city = db.Column(db.String(50), nullable=False)
    min_budget = db.Column(db.Float, nullable=False)
    required_facility = db.Column(db.String(100), nullable=False)
    created_at = db.Column(DateTime, default=datetime.now)

    user = db.relationship('User', backref=db.backref('search_queries', lazy=True))

@app.route('/hotel/<string:hotel_name>/reviews', methods=['GET', 'POST'])
def hotel_reviews(hotel_name):
    hotel_reviews = Review.query.filter_by(hotel_name=hotel_name).order_by(desc(Review.created_at)).all()

    if request.method == 'POST':
        rating =float(request.form['rating'])
        review_text = request.form['review_text']

        # Check if the user has already reviewed the hotel
        existing_review = Review.query.filter_by(user_id=session.get('user_id'), hotel_name=hotel_name).first()

        if existing_review:
            # Update existing review if found
            existing_review.rating = rating
            existing_review.review_text = review_text
        else:
            # Create a new review object
            new_review = Review(user_id=session['user_id'], hotel_name=hotel_name, rating=rating, review_text=review_text)
            db.session.add(new_review)

        # Commit changes to the database
        db.session.commit()

        # Update the hotel rating in google_hotel.csv
        matching_hotels = hotel_df[hotel_df["Hotel_Name"].str.lower() == hotel_name.lower()]

        if not matching_hotels.empty:
            hotel_row = matching_hotels.iloc[0]
            existing_ratings = hotel_row["Hotel_Rating"]
            if pd.notna(existing_ratings):
                existing_ratings = float(existing_ratings)
                new_avg_rating = round((existing_ratings + rating) / 2, 1)
            else:
                new_avg_rating = round(rating, 1)

            hotel_df.loc[hotel_df["Hotel_Name"].str.lower() == hotel_name.lower(), "Hotel_Rating"] = new_avg_rating

            # Save the updated DataFrame back to google_hotel.csv
            hotel_df.to_csv("google_hotel.csv", index=False)

        # Redirect to the same page after submitting the review to refresh the reviews
        return redirect(url_for('hotel_reviews', hotel_name=hotel_name))

    return render_template('hotel_reviews.html', hotel_name=hotel_name, reviews=hotel_reviews)

# Load the tourism data from Excel sheets
places_df = pd.read_csv("Places.csv")
city_df = pd.read_csv("City.csv")

# Attraction review database-->WORKS
class PlaceReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    place_name = db.Column(db.String(100), nullable=False)
    rating = db.Column(Numeric(precision=3, scale=1), nullable=False)
    review_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('place_reviews', lazy=True))

# Add this to your existing code
@app.route('/place/<string:place_name>/reviews', methods=['GET', 'POST'])
def place_reviews(place_name):
    place_reviews = PlaceReview.query.filter_by(place_name=place_name).order_by(desc(PlaceReview.created_at)).all()

    if request.method == 'POST':
        rating =float(request.form['rating'])
        review_text = request.form['review_text']

        # Check if the user has already reviewed the attraction
        existing_review = PlaceReview.query.filter_by(user_id=session.get('user_id'), place_name=place_name).first()

        if existing_review:
            # Update existing review if found
            existing_review.rating = rating
            existing_review.review_text = review_text
        else:
            # Create a new review object
            new_review = PlaceReview(user_id=session['user_id'], place_name=place_name, rating=rating, review_text=review_text)
            db.session.add(new_review)

        # Commit changes to the database
        db.session.commit()

        # new_review = PlaceReview(user_id=session['user_id'], place_name=place_name, rating=rating, review_text=review_text)
        # db.session.add(new_review)
        # db.session.commit()

        # Update the ratings for the specified place
        matching_places = places_df[places_df["Place"] == place_name]

        if not matching_places.empty:
          place_info = matching_places.iloc[0]
          existing_ratings = place_info["Ratings"]
          if pd.notna(existing_ratings):
            existing_ratings = float(existing_ratings)
            new_avg_rating = round((existing_ratings + rating) / 2, 1)
          else:
            new_avg_rating = round(rating, 1)

        places_df.loc[places_df["Place"] == place_name, "Ratings"] = new_avg_rating

        # Save the updated DataFrame back to Places.csv
        places_df.to_csv("Places.csv", index=False)

        # Redirect to the same page after submitting the review to refresh the reviews
        return redirect(url_for('place_reviews', place_name=place_name))

    return render_template('place_reviews.html', place_name=place_name, reviews=place_reviews)

# for using current user 
# print("load user is being called")
@login_manager.user_loader 
def load_user(id):
    print("User ID captured in load_user:", id)
    # Implement this function to return the User object for the given user_id
    return User.query.get(int(id))

# Home page with taskbar
@app.route('/home')
def home():
     return render_template('home.html')

@app.route('/beaches')
def beaches():
    return render_template('beaches.html')

@app.route('/hill-station')
def hillstation():
    return render_template('hill-station.html')

@app.route('/historical')
def historical():
    return render_template('historical.html')

@app.route('/parks')
def parks():
    return render_template('parks.html')

@app.route('/religious')
def religious():
    return render_template('religious.html')

@app.route('/honeymoon')
def honeymoon():
    return render_template('honeymoon.html')

@app.route('/map')
def map():
    return render_template('map.html')

# Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        age = request.form['age']
        gender = request.form['gender']
        country = request.form['country']
        city = request.form['city']
        area_code = request.form['area_code']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')

        # Check if username is unique
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error='Username already exists')
        
        # Check if the email is unique
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('register.html', error='Email already exists')

        # Hash the password before storing
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Add user to the database
        new_user = User(username=username, email=email,age=age, gender=gender, country=country, city=city, area_code=area_code, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Query the database for the user
        user = User.query.filter_by(username=username).first()

        # Check if the user exists and the password is correct
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store user ID in the session
            return redirect(url_for('home'))

        return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')


# Custom Jinja filters to strip numbers and brackets and fullstops
@app.template_filter('strip_numbers')
def strip_numbers(value):
    return ''.join(char for char in value if not char.isdigit())

@app.template_filter('strip_brackets')
def strip_brackets(value):
    return value.replace('[', '').replace(']', '')

@app.template_filter('strip_fullstop')
def strip_fullstop(value):
    return value.replace('.', '')


# unsplash API integration
def get_unsplash_image_url(place_name):
    unsplash_access_key = 'mE5I6IJm60NvhRKgL6nVNdVvPHAowbmPj2mpZLIXwCE'
    # below key the the one which runs
    # unsplash_url = f'https://api.unsplash.com/photos/random?query={place_name}&client_id={unsplash_access_key}'
    # office
    # unsplash_url = f'https://api.unsplash.com/search/photos?page=1&query={place_name}&client_id={unsplash_access_key}'

    # try:
    #     response = requests.get(unsplash_url)
    #     data = response.json()

    #     if 'urls' in data:
    #         return data['urls']['regular']

    # except Exception as e:
    #     print(f"Error fetching image for {place_name}: {e}")

    # return None


# recommend page
@app.route('/recommend')
def recommend():
    # Get the user's city from the session
    user_city = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        user_city = user.city

    # If user city is not available, provide a default city (you may modify this logic based on your application)
    if user_city is None:
        user_city = "Manali"

    # Retrieve city details from City.csv
    city_details = city_df[city_df['City'] == user_city].iloc[0]

    # Retrieve the top 5 places for the user's city from Places.csv
    top_places = places_df[places_df['City'] == user_city].nlargest(9, 'Ratings')


    # Get Unsplash image URLs for each place
    # top_places['image_url'] = top_places['Place'].apply(get_unsplash_image_url)

    # Render the recommend.html template with the recommendation details
    return render_template('recommend.html', 
                           city=user_city, 
                           ideal_duration=city_details['Ideal_duration'], 
                           best_time_to_visit=city_details['Best_time_to_visit'], 
                           city_desc=city_details['City_desc'], 
                           top_places=top_places.to_dict(orient="records"))

# access key
# mE5I6IJm60NvhRKgL6nVNdVvPHAowbmPj2mpZLIXwCE

#secret key
# G7iNe35hYOQZlzlHfbJh5efRUIG07e0aJPgI9idi4m4

#outer main page
@app.route('/')
def outer():
    # Your implementation for the recommend route
    return render_template('outer.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Load the datasets
hotel_df = pd.read_csv("google_hotel.csv")
cities_df = pd.read_csv("cities_in_india.csv")
city_info_df = pd.read_csv("City.csv")

@app.route("/index")
def index():
    # Render the index.html template with options for state and city
    states = cities_df["State"].unique()
    return render_template("index.html", states=states)


@app.route("/result", methods=["POST"])
# @login_required
def result():
    if request.method == "POST":

        # Get user inputs from the form
        state = request.form["state"].lower()
        city = request.form["city"].lower()
        min_budget = float(request.form["min_budget"])
        required_facility = request.form["required_facility"].lower()  # Convert to lowercase for case-insensitive comparison

        # code for collabortive/content based starts
        # print("Hello world")

        # Check if the same search query by the user exists in the database within a certain time frame 
        existing_query = SearchQuery.query.filter_by(
                user_id = session.get('user_id'),
                state=state,
                city=city,
                min_budget=min_budget,
                required_facility=required_facility
            ).filter(SearchQuery.created_at >= (datetime.now() - timedelta(hours=120))).first()
        # print(existing_query)

        if existing_query:
            # Update existing query if found
                existing_query.created_at = datetime.now()
                # print("Updated")
        else:
            # Save captured information in the database (e.g., SearchQuery table)
                new_query = SearchQuery(
                  user_id = session.get('user_id'),
                  state=state,
                  city=city,
                  min_budget=min_budget,
                  required_facility=required_facility
                 )
                db.session.add(new_query)
                # print("Saved as new entry")

        db.session.commit()
        # code for colleborative/content based ends            




        # Filter hotels based on city
        city_hotels = hotel_df[hotel_df["City"].str.lower() == city]

        # right one
        matching_facility = city_hotels[city_hotels.apply(
        lambda row: any(required_facility in str(row[feature]).lower() for feature in ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9']),
        axis=1)]
        

        # Filter hotels based on budget
        filtered_hotels = matching_facility[matching_facility["Hotel_Price"] <= min_budget]      

        # Get top 5 hotels with max ratings
        top_hotels = filtered_hotels.nlargest(6, "Hotel_Rating")
        # print(city)
        # print("TOP HOTELS:")
        # print(top_hotels)        

        matching_cities = city_info_df[city_info_df["City"].str.lower()  == city]

        if not matching_cities.empty:
          city_info = matching_cities.iloc[0]
          best_time_to_visit = city_info["Best_time_to_visit"]
          city_desc = city_info["City_desc"]



        # Check if best_time_to_visit is NaN and set it to "All year round" in that case
          if pd.isna(best_time_to_visit):
            best_time_to_visit = "All year round"
        #   print(best_time_to_visit)
        else:
            # If no matching city is found, set default values or leave them empty
            best_time_to_visit = "All year round" 
            city_desc = None


        #for attractions
        # Load the Places.csv file into a DataFrame

        # Filter places based on the selected city
        city_places = places_df[places_df["City"].str.lower() == city]

        # Sort places by Ratings in descending order
        sorted_places = city_places.sort_values(by="Ratings", ascending=False)

        # Select the top 5 places
        top_places = sorted_places.head(5)

         # Render the result.html template with the filtered hotels and city information 
        return render_template(
            "result.html",
            state=state,
            city=city,
            best_time_to_visit=best_time_to_visit,
            city_desc=city_desc,
            hotels=top_hotels.to_dict(orient="records"),
            top_places=top_places.to_dict(orient="records"),
)


# Reviews about city-->WORKS
class CityReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    city = db.Column(db.String(50), nullable=False)
    rating = db.Column(Numeric(precision=3, scale=1), nullable=False)
    review_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('city_reviews', lazy=True))


@app.route('/city/<string:city>/add_review', methods=['GET', 'POST'])
def add_city_reviews(city):
    city_reviews = CityReview.query.filter_by(city=city).order_by(desc(CityReview.created_at)).all()

    if request.method == 'POST':
        rating = float(request.form['rating'])
        review_text = request.form['review_text']

        existing_review = CityReview.query.filter_by(user_id=session.get('user_id'), city=city).first()

        if existing_review:
            # If an existing review is found, update its attributes
            existing_review.rating = rating
            existing_review.review_text = review_text
        else:
            # If no existing review is found, create a new one
            new_review = CityReview(user_id=session.get('user_id'), city=city, rating=rating, review_text=review_text)
            db.session.add(new_review)

        db.session.commit()
        # new_review = CityReview(user_id=session['user_id'], city=city, rating=rating, review_text=review_text)
        # db.session.add(new_review)
        # db.session.commit()

        # Update the city ratings in City.csv (Assuming city_info_df is a DataFrame loaded from City.csv)
        matching_cities = city_info_df[city_info_df["City"].str.lower() == city.lower()]

        if not matching_cities.empty:
            city_info = matching_cities.iloc[0]
            existing_ratings = city_info["Ratings"]
            if pd.notna(existing_ratings):
                existing_ratings = float(existing_ratings)
                new_avg_rating = round((existing_ratings + rating) / 2, 1)
            else:
                new_avg_rating = round(rating, 1)

            city_info_df.loc[city_info_df["City"].str.lower() == city.lower(), "Ratings"] = new_avg_rating

            # Save the updated DataFrame back to City.csv
            city_info_df.to_csv("City.csv", index=False)

        # Redirect to the same page after submitting the review to refresh the reviews
        return redirect(url_for('view_city_reviews', city=city))

    return render_template('result.html', city=city, city_reviews=city_reviews)


# Add this new route at the bottom of your app.py
@app.route('/city/<string:city>/reviews', methods=['GET'])
def view_city_reviews(city):
    city_reviews = CityReview.query.filter_by(city=city).order_by(desc(CityReview.created_at)).all()
    return render_template('city_reviews.html', city=city, city_reviews=city_reviews)


# Add this route for deleting a review
@app.route('/delete_review/<int:review_id>', methods=['POST'])
def delete_review(review_id):
    review_to_delete = CityReview.query.get(review_id)

    if review_to_delete:
        db.session.delete(review_to_delete)
        db.session.commit()

    # Redirect back to the same page after deleting the review
    return redirect(request.referrer)










# gemini
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse import hstack

# # Load hotel and attraction data (replace with your actual loading logic)

# hotels_df = pd.read_csv("hot.csv")
# hotels_df.fillna("", inplace=True)

# # print("Dataype of hotels infor is:",hotels_df.dtypes)
# # missing_values = hotels_df.isnull().sum()
# # print(missing_values[missing_values > 0])  # Print columns with missing values

# attractions_df = pd.read_csv("att.csv")
# attractions_df.fillna("", inplace=True) 
# attractions_df["Place_desc"].fillna("", inplace=True)  # Replace NaN values with empty strings


# # Function to preprocess text data (e.g., hotel descriptions, attraction details)
# def preprocess_text(text):
#     # Apply text cleaning steps here (e.g., lowercase, remove punctuation, stopwords)
#     if isinstance(text, str):
#         return text.lower().strip()
#     else:
#         return text    

# # Preprocess hotel and attraction descriptions
# hotels_df["Hotel_Name"] = hotels_df["Hotel_Name"].apply(preprocess_text)
# hotels_df["City"] = hotels_df["City"].apply(preprocess_text)
# hotels_df["Feature_1"] = hotels_df["Feature_1"].apply(preprocess_text)
# hotels_df["Feature_2"] = hotels_df["Feature_2"].apply(preprocess_text)
# hotels_df["Feature_3"] = hotels_df["Feature_3"].apply(preprocess_text)
# hotels_df["Feature_4"] = hotels_df["Feature_4"].apply(preprocess_text)
# hotels_df["Feature_5"] = hotels_df["Feature_5"].apply(preprocess_text)
# hotels_df["Feature_6"] = hotels_df["Feature_6"].apply(preprocess_text)
# hotels_df["Feature_7"] = hotels_df["Feature_7"].apply(preprocess_text)
# hotels_df["Feature_8"] = hotels_df["Feature_8"].apply(preprocess_text)
# hotels_df["Feature_9"] = hotels_df["Feature_9"].apply(preprocess_text)
# attractions_df["City"] = attractions_df["City"].apply(preprocess_text)
# attractions_df["Place_desc"] = attractions_df["Place_desc"].apply(preprocess_text)

# hotels_df_unique = hotels_df.groupby('City').first().reset_index()
# print(hotels_df_unique)
# # Create TF-IDF vectors for hotel and attraction descriptions
# # Assuming preprocessed text is stored in separate columns

# hotel_vectorizer = TfidfVectorizer(max_features=1000)
# hotel_text_features = ["Hotel_Name", "City"] + ["Feature_" + str(i) for i in range(1, 10)]
# hotels_df[hotel_text_features[1:]] = hotels_df[hotel_text_features[1:]].astype(str).map(preprocess_text)

# for col in hotel_text_features:
#     hotels_df[col] = hotels_df[col].fillna("").astype(str).apply(preprocess_text)

# hotels_df_unique = hotels_df.groupby('City').first().reset_index()
# hotels_df['combined_feat'] = hotels_df.apply(lambda row: ' '.join(row[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9']]), axis=1)
# print("Combined Features:")
# print(hotels_df['combined_feat'])

# hotel_features_matrix = pd.DataFrame()

# # Assuming you decide on a max_length of 100 words
# max_len = 100
# hotels_df[hotel_text_features] = hotels_df[hotel_text_features].apply(lambda x: x[:max_len] if len(x) > max_len else x.str.pad(max_len))


# for feature in hotel_text_features:
#     try:
#         print("Processing feature:", feature)
#         sparse_matrix = hotel_vectorizer.fit_transform(hotels_df[feature])
#         dense_array = sparse_matrix.toarray()
#         print("hotels_df shape:", hotels_df.shape)
#         print(f"Shape of dense_array for feature '{feature}':", dense_array.shape)    
#         hotel_features_matrix[feature] = dense_array.ravel()  # Reshape to 1D
#     except ValueError as e:
#         print("Error encountered for feature:", feature)
#         print(e)
#         # print(hotels_df[feature].head())  # Print a sample of values
#         # print(hotels_df[feature].isnull().sum())  # Check for missing values
#         # print(hotels_df[feature].dtype)  # Verify data type


# # Concatenate features into a single string (optional)
# hotels_df["combined_features"] = hotels_df[hotel_text_features].apply(lambda x: ' '.join(x), axis=1)
# combined_features_matrix = hotel_vectorizer.fit_transform(hotels_df["combined_features"])


# attractions_df["City"] = attractions_df["City"].fillna("").astype(str).apply(preprocess_text)
# attractions_df["Place_desc"] = attractions_df["Place_desc"].fillna("").astype(str).apply(preprocess_text)
# attraction_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
# attraction_text_features = ["City", "Place_desc"]

# attraction_features_matrix = attraction_vectorizer.fit_transform(attractions_df[attraction_text_features])









# def create_user_profile_vector(user_id):
#     # Fetch user data from database models
#     user_data = User.query.filter_by(id=user_id).first()
#     user_search_query = SearchQuery.query.filter_by(user_id=user_id).first()

#     user_profile = {}

#     # Extract relevant features from user_data and user_search_query
#     # Convert categorical features to numerical representations if needed

#     # Basic features from User model:
#     if user_data.age is not None:
#         user_profile["age"] = user_data.age
#     if user_data.gender is not None:
#         # Consider one-hot encoding or label encoding for gender
#         user_profile["gender"] = user_data.gender
#     # ... (add other relevant features from User model)

#     # Features from SearchQuery model:
#     if user_search_query is not None:
#         user_profile["city"] = user_search_query.city
#         user_profile["required_facilitiy"] = user_search_query.required_facility.split(",")  # Assuming facilities are comma-separated

#     # One-hot encode categorical features (replace with your chosen features)
#     categorical_features = ['gender']  # Adjust as needed
#     categorical_features_search = ['city','required_facility'] 

#     encoder = OneHotEncoder()

#     encoded_data = encoder.fit_transform([[getattr(user_data, feat) for feat in categorical_features]])
#     encoded_data_search = encoder.fit_transform([[getattr(user_search_query, feat) for feat in categorical_features_search]])

#     # Merge the encoded sparse matrices horizontally
#     user_profile_encoded = hstack([encoded_data, encoded_data_search])

#     user_profile["encoded_features"] = user_profile_encoded

#     print("User profile is:",user_profile)
#     return user_profile


# def generate_recommendations(user_profile, hotels_df, attractions_df, k=5):
#     # Calculate cosine similarity between user profile and hotels/attractions
#     hotel_similarities = cosine_similarity(user_profile["encoded_features"], hotel_features_matrix)
#     attraction_similarities = cosine_similarity(user_profile["encoded_features"], attraction_features_matrix)

#     # Combine similarities if needed for a hybrid approach
#     # ... (implementation for merging scores)

#     # Get indices of top k most similar hotels and attractions
#     top_k_hotel_indices = hotel_similarities.argsort()[-k:][::-1]
#     top_k_attraction_indices = attraction_similarities.argsort()[-k:][::-1]

#     # Retrieve recommended items from the DataFrames
#     recommended_hotels = hotels_df.iloc[top_k_hotel_indices]
#     recommended_attractions = attractions_df.iloc[top_k_attraction_indices]

#     return recommended_hotels, recommended_attractions

# @app.route('/example')
# def example_route():
#     user_id = session.get('user_id')
#     user_profile = create_user_profile_vector(user_id)
#     # Generate recommendations
#     recommended_hotels, recommended_attractions = generate_recommendations(user_profile, hotels_df, attractions_df)

#     # Pass recommendations to the template for display    
#     return render_template('home.html', recommended_hotels=recommended_hotels, recommended_attractions=recommended_attractions)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from scipy import sparse as sp


# Load hotel data
hotels_df = pd.read_csv("hot.csv")
hotels_df.fillna("", inplace=True)

# Function to preprocess text data
def preprocess_text(text):
    if isinstance(text, str):
        return text.lower().strip()
    else:
        return text    

# Preprocess hotel data
for col in hotels_df.columns:
    hotels_df[col] = hotels_df[col].apply(preprocess_text)

# Group by City and combine features into a single column
hotels_df['combined_feat'] = hotels_df.apply(lambda row: ' '.join(row[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9']]), axis=1)
hotels_df_grouped = hotels_df.groupby('City').first().reset_index()

# Create TF-IDF vectors for combined features
vectorizer = TfidfVectorizer(max_features=1000)
hotel_features_matrix = vectorizer.fit_transform(hotels_df_grouped['combined_feat'])

from scipy.sparse import hstack, vstack

def create_user_profile_vector(user_id):
    # Fetch user data from database models
    user_data = User.query.filter_by(id=user_id).first()
    user_search_queries = SearchQuery.query.filter_by(user_id=user_id).all()  # Fetch all search queries for the user

    user_profile = {}

    # Extract relevant features from user_data and user_search_queries
    # Basic features from User model:
    if user_data:
        if user_data.age is not None:
            user_profile["age"] = user_data.age
        if user_data.gender is not None:
            user_profile["gender"] = user_data.gender

    # Features from SearchQuery model:
    cities = []
    facilities = []
    for search_query in user_search_queries:
        cities.append(search_query.city)
        if search_query.required_facility:
            facilities.extend(search_query.required_facility.split(","))
            
    user_profile["city"] = cities

    # One-hot encode categorical features
    categorical_features = ['gender']  # Adjust as needed
    encoder = OneHotEncoder()

    encoded_data = encoder.fit_transform([[user_profile.get(feat, None)] for feat in categorical_features])
    
    # Convert required_facility to binary features
    mlb = MultiLabelBinarizer()
    facilities_binary = mlb.fit_transform([[facility] for facility in facilities])
    user_profile["required_facility_binary"] = facilities_binary

    # Ensure both matrices have the same number of rows before stacking horizontally
    max_rows = max(encoded_data.shape[0], user_profile["required_facility_binary"].shape[0])
    encoded_data_padded = vstack([encoded_data, sp.csr_matrix((max_rows - encoded_data.shape[0], encoded_data.shape[1]))])
    required_facility_binary_padded = vstack([user_profile["required_facility_binary"], sp.csr_matrix((max_rows - user_profile["required_facility_binary"].shape[0], user_profile["required_facility_binary"].shape[1]))])

    # Stack matrices horizontally
    user_profile_encoded = hstack([encoded_data_padded, required_facility_binary_padded])

    user_profile["encoded_features"] = user_profile_encoded

    print("Encoded data shape:", encoded_data.shape)
    print("Facility binary shape:", user_profile["required_facility_binary"].shape)
    print("User profile is:", user_profile)

    return user_profile



  # Separate user features and facilities (assuming facilities are binary):

# Define generate_recommendations function (modify as needed)
def generate_recommendations(user_profile, hotels_df, k=5):

    # Calculate cosine similarity between user profile and hotels
    user_vector = user_profile["encoded_features"]
    # Inside the generate_recommendations function, after fetching user_vector:
    user_vector = user_vector.reshape(1, -1)  # Reshape to 1 row and 20 columns

    print("Shape of user_vector:", user_vector.shape)
    print("Shape of hotel_features_matrix:", hotel_features_matrix.shape)

    # Print the first few rows of the user vector and hotel features matrix
    print("User vector:")
    print(user_vector.tocsr()[:2])  # Print the first 5 rows
  # Print the first 5 rows
    print("Hotel features matrix:")
    print(hotel_features_matrix[:2])  # Print the first 5 rows

    # Inside the generate_recommendations function
    if user_vector.shape[0] == 0 or hotel_features_matrix.shape[0] == 0:
        print("User profile or hotel features matrix is empty.")
        # Handle empty matrices, e.g., return default recommendations
        return None


    cosine_similarities = cosine_similarity(user_vector, hotel_features_matrix)

    # Get indices of top-k most similar hotels
    top_k_indices = cosine_similarities.argsort()[:, ::-1][:, :k]

    # Retrieve recommended hotels from the DataFrames
    recommended_hotels = hotels_df_grouped.iloc[top_k_indices[0]]

    return recommended_hotels

# Assuming you have Flask routes defined below

@app.route('/example')
def example_route():
    user_id = session.get('user_id')
    user_profile = create_user_profile_vector(user_id)
    recommended_hotels = generate_recommendations(user_profile, hotels_df_grouped)
    return render_template('home.html', recommended_hotels=recommended_hotels)























# @app.route('/home_recommend')
# def home_recommend():

#     # Load hotel dataset 
#     # Drop irrelevant columns (images)
#     hotel_data = hotel_df
#     hotel_data = hotel_data.drop(columns=['Image','Hotel_Price','Hotel_Rating'])
#     attraction_data = places_df
#     # attraction_data = attraction_data.drop(columns=['Images'])
#     attraction_data = attraction_data.drop(columns=['Images', 'Distance', 'Place_desc','Ratings'])

#     # Convert text fields to lowercase
#     hotel_data['Hotel_Name'] = hotel_data['Hotel_Name'].str.lower()
#     hotel_data['City'] = hotel_data['City'].str.lower()
#     for feature in ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9']:
#      hotel_data[feature] = hotel_data[feature].str.lower()
#     attraction_data['Place'] = attraction_data['Place'].str.lower()
#     attraction_data['City'] = attraction_data['City'].str.lower()

#     # Save cleaned and normalized datasets
#     hotel_data.to_csv('cleaned_hotel_dataset.csv', index=False)
#     attraction_data.to_csv('cleaned_attraction_dataset.csv', index=False)

#     # Fetch user instances,search query instances from the database
#     users = User.query.all()
#     search_queries = SearchQuery.query.all()

#     # user vector
#     user_vectors = create_user_vectors(users, search_queries)
#     print("User vector type:", type(user_vectors))
#     print("User vector content:", user_vectors)

#     # for similarity calculation
#     vector_for_user=vectorize_user(user_vectors)
#     print("vectorized user vector:",vector_for_user)


#     # Load cleaned hotel and attraction datasets from CSV files
#     hotel_da = pd.read_csv('cleaned_hotel_dataset.csv')

#     # Initialize OneHotEncoder
#     encoder = OneHotEncoder()

#     # Fit and transform the city data
#     city_encoded = encoder.fit_transform(hotel_da[['City']])

#     # Convert to DataFrame
#     city_encoded_df = pd.DataFrame(city_encoded.toarray(), columns=encoder.get_feature_names_out(['City']))

#     # Concatenate with existing features
#     hotel_data_encoded = pd.concat([hotel_da.drop(columns=['City']), city_encoded_df], axis=1)

#     attraction_da = pd.read_csv('cleaned_attraction_dataset.csv')

#     # Convert hotel and attraction data to dictionary format

#     attractions = []
#     hotels = []
#     for index, row in hotel_da.iterrows():
#         hotel = {
#             'name': row['Hotel_Name'],
#             'city': row['City'],
#             'city_encoded': city_encoded_df.iloc[index].values,  # Add city_encoded feature
#             'feature_1': row['Feature_1'],
#             'feature_2': row['Feature_2'],
#             'feature_3': row['Feature_3'],
#             'feature_4': row['Feature_4'],
#             'feature_5': row['Feature_5'],
#             'feature_6': row['Feature_6'],
#             'feature_7': row['Feature_7'],
#             'feature_8': row['Feature_8'],
#             'feature_9': row['Feature_9'],                
#         }
#         hotels.append(hotel)
    
#     for index, row in attraction_da.iterrows():
#         attraction = {
#             'name': row['Place'],
#             'city': row['City'],
#         }
#         attractions.append(attraction)


#     hotel_vectors = [vectorize_hotel(hotel) for hotel in hotels]
#     attraction_vectors = [vectorize_attraction(attraction) for attraction in attractions]
#     print("Shape of hotel_vectors:", np.array(hotel_vectors).shape)
#     print("Shape of attraction_vectors:", np.array(attraction_vectors).shape)


#     item_item_similarities_hotels = calculate_item_item_similarity(hotel_vectors)
#     item_item_similarities_attractions = calculate_item_item_similarity(attraction_vectors)  

#     # user_item_similarities_hotels = calculate_user_item_similarity(vector_for_user, hotel_vectors)
#     # user_item_similarities_attractions = calculate_user_item_similarity(vector_for_user, attraction_vectors)   

#     # Render the home.html template with the recommendation results
#     return render_template('home.html')



# def create_user_vectors(users, search_queries):
#     user_vectors = defaultdict(lambda: defaultdict(int))

#     # Iterate over search queries
#     for query in search_queries:
#         user_id = query.user_id
#         city = query.city
#         required_facility = query.required_facility

#         # Increment the count of the city and required facility for the user
#         user_vectors[user_id]['city_' + city] += 1
#         user_vectors[user_id]['facility_' + required_facility] += 1

#     return user_vectors

# # Convert user vector to a format suitable for cosine similarity calculation
# # def vectorize_user(user_vector):
# #     vectorized_users = []

# #     for inner_dict in user_vector.values():
# #         vectorized_user = list(inner_dict.values())
# #         vectorized_users.append(vectorized_user)
# #     return vectorized_users

# def vectorize_user(user_vector):
#     vectorized_users = []
#     print("User vector type:", type(user_vector))
#     print("User vector content:", user_vector)
#     for inner_dict in user_vector.values():
#         vectorized_user = [inner_dict[key] for key in sorted(inner_dict.keys())]
#         vectorized_users.append(vectorized_user)
#     return vectorized_users


# # Convert hotel and attraction vectors to a format suitable for cosine similarity calculation
# def vectorize_hotel(item):
#     return [
#             *item['city_encoded'],
#             item.get('feature_1', 0),
#             item.get('feature_2', 0),
#             item.get('feature_3', 0),
#             item.get('feature_4', 0),
#             item.get('feature_5', 0),
#             item.get('feature_6', 0),
#             item.get('feature_7', 0),
#             item.get('feature_8', 0),
#             item.get('feature_9', 0),                          
#             ]  

# def vectorize_attraction(attraction):
#     return [attraction.get('rating', 0),
#             attraction.get('city', '')
#             ]

# # Calculate item-item similarity
# # def calculate_item_item_similarity(item_vectors):
# #     item_similarities = cosine_similarity(item_vectors)
# #     return item_similarities


# # Ensure input format and dimensions are correct for cosine_similarity
# def calculate_item_item_similarity(item_vectors):
#     # Convert item_vectors to numpy array to ensure consistency
#     item_vectors_np = np.array(item_vectors)
#     # Check if item_vectors_np is a 2D array
#     if item_vectors_np.ndim != 2:
#         raise ValueError("Input item_vectors must be a 2D array.")
#     # Check if all rows have the same length
#     if len(set(len(row) for row in item_vectors_np)) != 1:
#         raise ValueError("All rows in item_vectors must have the same length.")
#     # Verify the validity of numerical data
#     if np.isnan(item_vectors_np).any() or not np.isfinite(item_vectors_np).all():
#         raise ValueError("Input item_vectors contains NaN or infinite values.")
#     # Calculate cosine similarity
#     item_similarities = cosine_similarity(item_vectors_np)
#     return item_similarities

# # The rest of your code remains unchanged





if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables before running the app
    app.run(debug=True)













    # Define features to be normalized
    # features_to_normalize_hotel = ['Hotel_Rating', 'Hotel_Price']
    # features_to_normalize_attraction = ['Ratings']

    # Initialize MinMaxScaler
    # scaler = MinMaxScaler()

    # # Normalize features for hotel dataset
    # hotel_data[features_to_normalize_hotel] = scaler.fit_transform(hotel_data[features_to_normalize_hotel])

    # # Normalize features for attraction dataset
    # attraction_data[features_to_normalize_attraction] = scaler.fit_transform(attraction_data[features_to_normalize_attraction])
















# print("User vectors after search queries processing:")
    # print(user_vectors)

    # Add age and gender from User table
    # for user in users:
    #     user_id = user.id
    #     age = user.age
    #     gender = user.gender

    #     # Add age and gender to the user vector
    #     user_vectors[user_id]['age'] = age
    #     user_vectors[user_id]['gender_' + gender] = 1

    # print("User vectors after adding age and gender:")
    # print(user_vectors)

    # Normalize user vectors
    # for user_id, vector in user_vectors.items():
    #     total_queries = sum(vector.values())
    #     for key in vector:
    #         vector[key] /= total_queries

    # print("Normalized user vectors:")
    # print(user_vectors)