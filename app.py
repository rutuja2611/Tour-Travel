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



# collaborative and content based filtering
@app.route('/home_recommend')
def home_recommend():
    # Data Cleaning and Preprocessing for Hotel Data
    # Drop rows with missing values in any column except 'Image'
    hotel_df_cleaned = hotel_df.dropna(subset=[col for col in hotel_df.columns if col != 'Image'])
    hotel_df_cleaned['Hotel_Rating'] = hotel_df_cleaned['Hotel_Rating'].astype(float)

    # Data Cleaning and Preprocessing for Attraction Data
    # Drop rows with missing values in any column except 'Image' for places_df
    places_df_cleaned = places_df.dropna(subset=[col for col in places_df.columns if col != 'Images'])
    places_df_cleaned['Ratings'] = places_df_cleaned['Ratings'].astype(float)

    # Convert city names to lowercase
    hotel_df_cleaned['City'] = hotel_df_cleaned['City'].str.lower()
    places_df_cleaned['City'] = places_df_cleaned['City'].str.lower()

    # Feature Encoding for Hotel Data (One-Hot Encoding for City)
    hotel_df_encoded = pd.get_dummies(hotel_df_cleaned, columns=['City'])

    # Feature Encoding for Attraction Data (One-Hot Encoding for City)
    places_df_encoded = pd.get_dummies(places_df_cleaned, columns=['City'])

    # Min-Max Scaling for Hotel Price
    scaler = MinMaxScaler()
    hotel_df_encoded['Hotel_Price'] = scaler.fit_transform(hotel_df_encoded[['Hotel_Price']])

     # Feature Extraction for Hotel features
    hotel_features = hotel_df_cleaned[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9']]

    # this consider only features
    hotel_feature_vector = pd.get_dummies(hotel_features)

    # Feature Extraction for Attraction description
    attraction_feature_vector = pd.get_dummies(places_df_cleaned['Place_desc'])

    user_id = int(session.get('user_id'))
    user = db.session.query(User).get(user_id)
    age = user.age
    gender = user.gender

    # Retrieve all search preferences of the user from the database
    user_queries = None
    user_queries = SearchQuery.query.filter_by(user_id = session.get('user_id')).all()   

    # Proceed if user_queries exist
    if user_queries:
        # Initialize lists to store search preferences
        states = []
        cities = []
        min_budgets = []
        required_facilities = []

        # Extract search preferences from user_queries
        for query in user_queries:
            states.append(query.state)
            cities.append(query.city)
            min_budgets.append(query.min_budget)
            required_facilities.append(query.required_facility)


    # Retrieve user ratings for cities  
    city_ratings = CityReview.query.filter_by(user_id=user_id).with_entities(CityReview.city, CityReview.rating).all()


    # Retrieve user ratings for hotels
    hotel_ratings = Review.query.filter_by(user_id=user_id).with_entities(Review.hotel_name, Review.rating).all()

    # Retrieve user ratings for attractions
    attraction_ratings = PlaceReview.query.filter_by(user_id=user_id).with_entities(PlaceReview.place_name, PlaceReview.rating).all()        

    # Proceed with creating user feature vector using demographic information and all search preferences
    user_feature_vector = create_user_feature_vector(age, gender, states, cities, min_budgets, required_facilities, city_ratings, hotel_ratings, attraction_ratings)


    #combine all features of hotel-> final hotel feature vector
    city_columns = hotel_df_encoded.filter(like='City_').columns
    hotel_features_combined = pd.concat([hotel_df_encoded[city_columns], hotel_df_encoded[['Hotel_Rating', 'Hotel_Price']], hotel_feature_vector], axis=1)
      # did nto understand below step
    hotel_features_combined.fillna(hotel_features_combined.mean(), inplace=True)


    #combine all the features of attractions-->final attraction feature vector
    city_columns = places_df_encoded.filter(like='City_').columns
    attraction_features_combined = pd.concat([places_df_encoded[city_columns], places_df_encoded[['Ratings']], attraction_feature_vector], axis=1)
    # did nto understand below step
    attraction_features_combined.fillna(attraction_features_combined.mean(), inplace=True)



    #content based filtering--> calculating similarities betwwen item-item and user-item to provide recommendations

    #similarity matrix item-item similarity
    hotel_similarity_matrix,attraction_similarity_matrix = calculate_item_item_similarity(hotel_features_combined,attraction_features_combined)

    # user-item similairty
    user_item_hotel_similarity = calculate_user_item_hotel_similarity(user_feature_vector,hotel_df_encoded, hotel_features_combined)
    user_attraction_similarity = calculate_user_item_attractions_similarity(user_feature_vector, places_df_encoded, attraction_features_combined)

    # Collaborative Filtering
    hotel_cf_recommendations = collaborative_filtering_hotel_recommendation(user_id, hotel_df_encoded, hotel_similarity_matrix)
    attraction_cf_recommendations = collaborative_filtering_attractions_recommendation(user_id, places_df_encoded, attraction_similarity_matrix)
    

    # Render the home.html template with the recommendation results
    return render_template('home.html', hotel_recommendations=hotel_cf_recommendations, attraction_recommendations=attraction_cf_recommendations)
            # , recommendation_results=recommendation_results


def create_user_feature_vector(age, gender, states, cities, min_budgets, required_facilities, city_ratings, hotel_ratings, attraction_ratings):
    # Initialize feature vector
    user_feature_vector = {}

    # Demographic information
    user_feature_vector['age'] = age
    user_feature_vector['gender'] = gender

    # Search preferences
    user_feature_vector['states'] = states
    user_feature_vector['cities'] = cities
    user_feature_vector['min_budgets'] = min_budgets
    user_feature_vector['required_facilities'] = required_facilities

    # Ratings for cities
    for city, rating in city_ratings:
        user_feature_vector[f'city_rating_{city}'] = rating

    # Ratings for hotels
    for hotel, rating in hotel_ratings:
        user_feature_vector[f'hotel_rating_{hotel}'] = rating

    # Ratings for attractions
    for attraction, rating in attraction_ratings:
        user_feature_vector[f'attraction_rating_{attraction}'] = rating

    return user_feature_vector

# Item-Item Similarity Calculation
def calculate_item_item_similarity(hotel_features_combined,attraction_features_combined):

    # Calculate item-item similarity matrix for hotels
    hotel_similarity_matrix = cosine_similarity(hotel_features_combined)    
    # Calculate item-item similarity matrix for attractions
    attraction_similarity_matrix = cosine_similarity(attraction_features_combined)
    
    return hotel_similarity_matrix, attraction_similarity_matrix

def calculate_user_item_hotel_similarity(user_feature_vector,hotel_df_encoded, hotel_features_combined):

    # Calculate cosine similarity between user and each hotel
    # user_vector = np.array(list(user_feature_vector.values())).reshape(1, -1)
    user_vector = np.array([v for v in user_feature_vector.values()]).reshape(1, -1)
    hotel_vectors = np.array(hotel_features_combined.values)
    similarities = cosine_similarity(user_vector, hotel_vectors)

    # Create a dictionary to store hotel IDs and their corresponding similarity scores
    hotel_similarity = {hotel_id: similarity[0] for hotel_id, similarity in zip(hotel_df_encoded.index, similarities)}

    return hotel_similarity

def calculate_user_item_attractions_similarity(user_feature_vector, places_df_encoded, attraction_features_combined):
    # Calculate cosine similarity between user and each attraction
    # user_vector = np.array(list(user_feature_vector.values())).reshape(1, -1)
    user_vector = np.array([v for v in user_feature_vector.values()]).reshape(1, -1)
    attraction_vectors = np.array(attraction_features_combined.values)
    similarities = cosine_similarity(user_vector, attraction_vectors)

    # Create a dictionary to store attraction IDs and their corresponding similarity scores
    attraction_similarity = {attraction_id: similarity[0] for attraction_id, similarity in zip(places_df_encoded.index, similarities)}

    return attraction_similarity


def collaborative_filtering_hotel_recommendation(user_id, df_encoded, similarity_matrix):
    # Get user's ratings
    user_ratings = Review.query.filter_by(user_id=user_id).all()

    # Predict ratings for unrated items
    predicted_ratings = {}
    for item_id in df_encoded.index:
        if item_id not in [rating.item_id for rating in user_ratings]:
            similarity_scores = similarity_matrix[item_id]
            rated_items = [rating.item_id for rating in user_ratings]
            weighted_sum = 0
            sum_of_weights = 0
            for i in range(len(similarity_scores)):
                if df_encoded.index[i] in rated_items:
                    weight = similarity_scores[i]
                    rating = Review.query.filter_by(user_id=user_id, item_id=df_encoded.index[i]).first().rating
                    weighted_sum += weight * rating
                    sum_of_weights += abs(weight)
            if sum_of_weights > 0:
                predicted_ratings[item_id] = weighted_sum / sum_of_weights

    # Sort predicted ratings in descending order
    sorted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)

    # Return top recommendations
    top_recommendations = sorted_ratings[:10]  # Change 10 to the number of recommendations needed
    print("Top hotels are:")
    print(top_recommendations)
    return top_recommendations


def collaborative_filtering_attractions_recommendation(user_id, df_encoded, similarity_matrix):
    # Get user's ratings
    user_ratings = PlaceReview.query.filter_by(user_id=user_id).all()

    # Predict ratings for unrated items
    predicted_ratings = {}
    for item_id in df_encoded.index:
        if item_id not in [rating.item_id for rating in user_ratings]:
            similarity_scores = similarity_matrix[item_id]
            rated_items = [rating.item_id for rating in user_ratings]
            weighted_sum = 0
            sum_of_weights = 0
            for i in range(len(similarity_scores)):
                if df_encoded.index[i] in rated_items:
                    weight = similarity_scores[i]
                    rating = PlaceReview.query.filter_by(user_id=user_id, item_id=df_encoded.index[i]).first().rating
                    weighted_sum += weight * rating
                    sum_of_weights += abs(weight)
            if sum_of_weights > 0:
                predicted_ratings[item_id] = weighted_sum / sum_of_weights

    # Sort predicted ratings in descending order
    sorted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)

    # Return top recommendations
    top_recommendations = sorted_ratings[:10]  # Change 10 to the number of recommendations needed
    print("Top attractions are:")
    print(top_recommendations)
    return top_recommendations





if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables before running the app
    app.run(debug=True)













   # Combine feature vectors
    # combined_feature_vector = pd.concat([hotel_feature_vector, attraction_feature_vector], axis=1)

    # # Example: Average rating of hotels in the same city
    # average_rating_by_city = hotel_df_encoded.groupby('City')['Hotel_Rating'].mean().reset_index()
    # average_rating_by_city.rename(columns={'Hotel_Rating': 'Avg_Hotel_Rating_by_City'}, inplace=True)
    # hotel_df_encoded = hotel_df_encoded.merge(average_rating_by_city, on='City', how='left')
