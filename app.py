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

@app.route('/home', methods=['GET', 'POST'])
def home():
    # Assuming user_id is available in session or request context
    user_id = session.get('user_id')  # Example user ID, replace with actual user ID retrieval
    print(user_id)
    last_rated_hotel = db.session.query(Review.hotel_name).filter(Review.user_id == user_id, Review.rating > 4).order_by(Review.created_at.desc()).first()
    print(last_rated_hotel)
    if last_rated_hotel:
        last_rated_hotel = last_rated_hotel[0]  # Extract hotel name from tuple
        recommendations = recomm(last_rated_hotel)  # Pass last_rated_hotel to recomm function
    else:
        recommendations = []




    #collaborative model
    ratings = pd.read_csv("co.csv", index_col=0)
    print("Read CSV file successfully")
    print(ratings.head(2))  # Print the first 2 entries of ratings DataFrame

    ratings = ratings.dropna(thresh=1, axis=1).fillna(0)
    print("Preprocessing ratings data")
    print(ratings.head(3))

    def standardize(row):
        mean = row.mean()
        min_val = row.min()
        max_val = row.max()
        if max_val == min_val:
            return 0.0
        else:
            return (row - mean) / (max_val - min_val)

    df_std = ratings.apply(standardize)
    print("Standardized ratings data")
    print(df_std.head(3))

    item_similarity = cosine_similarity(df_std.T)
    print("Computed item similarity matrix")
    print(item_similarity)  # Print similarity matrix if needed
    item_similarity_df = pd.DataFrame(item_similarity, index=ratings.columns, columns=ratings.columns)
    print("Item similarity DataFrame:")
    print(item_similarity_df)  # Print item similarity DataFrame if needed


    def get_similar(movie_name, rating):
        rating_float = float(rating)  # Convert Decimal to float
        similar_score = item_similarity_df[movie_name] * (rating_float - 2.5)
        similar_score = similar_score.sort_values(ascending=False)
        return similar_score


    user_id = session.get('user_id')  # Assuming current_user is available
    user_ratings = Review.query.filter_by(user_id=user_id).all()
    print("Retrieved user ratings")

    user_four = [(review.hotel_name, review.rating) for review in user_ratings]
    print("User Four:",user_four)
    similar_scores = pd.DataFrame()

    for movie, rating in user_four:
        similar_scores = pd.concat([similar_scores, get_similar(movie, rating)])

    # Group the similarity scores by movie name and sum them up
    collab_recommendations = similar_scores.groupby(level=0).sum()

    # Sort the DataFrame by total similarity score in descending order
    # collab_recommendations = collab_recommendations.sort_values(ascending=False)

    # Print the sorted collab_recommendations
    print("Sorted collab_recommendations:")
    print(collab_recommendations)


    top_recommendations = collab_recommendations.head(5).index.tolist()     
    print("Generated top recommendations")
    print("Generated top recommendations:", top_recommendations)

    return render_template('home.html', last_rated_hotel=last_rated_hotel, recommendations=recommendations,top_recommendations=top_recommendations)

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


    # Render the recommend.html template with the recommendation details
    return render_template('recommend.html', 
                           city=user_city, 
                           ideal_duration=city_details['Ideal_duration'], 
                           best_time_to_visit=city_details['Best_time_to_visit'], 
                           city_desc=city_details['City_desc'], 
                           top_places=top_places.to_dict(orient="records"))

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



# content based filtering
from sklearn.feature_extraction.text import CountVectorizer
# Load data
movies = pd.read_csv('hot.csv')
# print("First two entries after loading data:")
# print(movies.head(2))  # Print first two entries

# Keep required columns and preprocess data
movies = movies[['Hotel_Name', 'City', 'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9']]
movies.fillna("", inplace=True)  # Fill missing values
movies['tags'] = movies.apply(lambda row: ' '.join(row), axis=1)
# print("\nFirst two entries after preprocessing data:")
# print(movies.head(2))  # Print first two entries

# Vectorize the text data
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
# print("Shape of vector array:", vector.shape)

# Calculate cosine similarity
similarity = cosine_similarity(vector)
# print("Shape of similarity array:", similarity.shape)

# Recommendation function
def recomm(movie):
    if movies.empty:
        return "No movies available for recommendation."
    
    filtered_movies = movies[movies['Hotel_Name'] == movie]
    if filtered_movies.empty:
        return "No movies found with the given name."
    
    index = filtered_movies.index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = [movies.iloc[i[0]] for i in distances[1:6]]
    return recommendations


# collaborative filtering

import pandas as pd
from app import app, db, User, Review

# creating the dataset

# Load hotel names from hot.csv
hotels_df = pd.read_csv('hot.csv')

# Create a DataFrame for storing ratings
ratings_df = pd.DataFrame(columns=[''] + list(hotels_df['Hotel_Name']))

# Retrieve user IDs within the Flask app context
with app.app_context():
    user_ids = [user.id for user in User.query.all()]

# Set hotel names as column names
ratings_df.columns = [''] + list(hotels_df['Hotel_Name'])

# Set user IDs in the first column
ratings_df[''] = ['id of user: ' + str(user_id) for user_id in user_ids]

# Set user IDs as index
ratings_df.set_index('', inplace=True)


# Retrieve ratings from reviews within the Flask app context
with app.app_context():
    for user_id in user_ids:
        user_reviews = Review.query.filter_by(user_id=user_id).all()
        for review in user_reviews:
            hotel_name = review.hotel_name
            rating = review.rating
            ratings_df.at['id of user: ' + str(user_id), hotel_name] = rating

# Save ratings to co.csv
ratings_df.to_csv('co.csv')





# #model

# @app.route('/collab')
# def collab():
#     ratings = pd.read_csv("co.csv", index_col=0)
#     ratings = ratings.dropna(thresh=1, axis=1).fillna(0)

#     def standardize(row):
#         mean = row.mean()
#         min_val = row.min()
#         max_val = row.max()
#         if max_val == min_val:
#             return 0.0
#         else:
#             return (row - mean) / (max_val - min_val)

#     df_std = ratings.apply(standardize)

#     item_similarity = cosine_similarity(df_std.T)
#     item_similarity_df = pd.DataFrame(item_similarity, index=ratings.columns, columns=ratings.columns)

#     def get_similar(movie_name, rating):
#         similar_score = item_similarity_df[movie_name] * (rating - 2.5)
#         similar_score = similar_score.sort_values(ascending=False)
#         return similar_score

#     user_id = session.get('user_id')  # Assuming current_user is available
#     user_ratings = Review.query.filter_by(user_id=user_id).all()

#     user_four = [(review.hotel_name, review.rating) for review in user_ratings]
#     similar_scores = pd.DataFrame()

#     for movie, rating in user_four:
#         similar_scores = pd.concat([similar_scores, get_similar(movie, rating)], ignore_index=True)

#     recommendations = similar_scores.sum().sort_values(ascending=False)
#     top_recommendations = recommendations.head(5).index.tolist()
#     return render_template('home.html', top_recommendations=top_recommendations)



if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables before running the app
    app.run(debug=True)
