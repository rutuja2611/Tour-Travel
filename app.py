from flask import Flask, render_template, redirect, request, session, url_for
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
# import requests
from datetime import datetime
from sqlalchemy import Numeric
from sqlalchemy import func, desc  # Import func and desc from SQLAlchemy


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # Use SQLite for simplicity
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    country = db.Column(db.String(50), nullable=False)
    city = db.Column(db.String(50), nullable=False)
    area_code = db.Column(db.String(10), nullable=False)
    password = db.Column(db.String(60), nullable=False)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # place_name = db.Column(db.String(100), nullable=False)
    hotel_name = db.Column(db.String(100), nullable=False)
    rating = db.Column(Numeric(precision=3, scale=1), nullable=False)
    review_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('reviews', lazy=True))

@app.route('/hotel/<string:hotel_name>/reviews', methods=['GET', 'POST'])
def hotel_reviews(hotel_name):
    hotel_reviews = Review.query.filter_by(hotel_name=hotel_name).order_by(desc(Review.created_at)).all()

    if request.method == 'POST':
        rating =float(request.form['rating'])
        review_text = request.form['review_text']

        new_review = Review(user_id=session['user_id'], hotel_name=hotel_name, rating=rating, review_text=review_text)
        db.session.add(new_review)
        db.session.commit()

        # Redirect to the same page after submitting the review to refresh the reviews
        return redirect(url_for('hotel_reviews', hotel_name=hotel_name))

    return render_template('hotel_reviews.html', hotel_name=hotel_name, reviews=hotel_reviews)

# Load the tourism data from Excel sheets
places_df = pd.read_csv("Places.csv")
city_df = pd.read_csv("City.csv")

# Add this to your existing code
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

        new_review = PlaceReview(user_id=session['user_id'], place_name=place_name, rating=rating, review_text=review_text)
        db.session.add(new_review)
        db.session.commit()

        # Redirect to the same page after submitting the review to refresh the reviews
        return redirect(url_for('place_reviews', place_name=place_name))

    return render_template('place_reviews.html', place_name=place_name, reviews=place_reviews)




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
        new_user = User(username=username, email=email, country=country, city=city, area_code=area_code, password=hashed_password)
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
def result():
    if request.method == "POST":


        # Get user inputs from the form
        state = request.form["state"].lower()
        city = request.form["city"].lower()
        min_budget = float(request.form["min_budget"])
        required_facility = request.form["required_facility"].lower()  # Convert to lowercase for case-insensitive comparison

        # Filter hotels based on city
        city_hotels = hotel_df[hotel_df["City"].str.lower() == city]
        # print("City Hotels:")
        # print(city_hotels)

        # print("Required Facility:")
        # print(required_facility)

        # Filter hotels based on required facility
        # matching_facility = city_hotels[city_hotels.apply(lambda row: any(required_facility in str(row[feature]).lower() for feature in ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9']), axis=1)]

        # right one
        matching_facility = city_hotels[city_hotels.apply(
        lambda row: any(required_facility in str(row[feature]).lower() for feature in ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9']),
        axis=1)]
        
        # Split user input into a list of requirements

        # print("Matching Facility:")
        # print(matching_facility)


        # Filter hotels based on budget
        filtered_hotels = matching_facility[matching_facility["Hotel_Price"] <= min_budget]
        # print("HOTEL IN BUDGET:")
        # print(filtered_hotels)        

        # Get top 5 hotels with max ratings
        top_hotels = filtered_hotels.nlargest(6, "Hotel_Rating")
        print(city)
        print("TOP HOTELS:")
        print(top_hotels)        

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
        # print("city")
        # print(city)

        # Sort places by Ratings in descending order
        sorted_places = city_places.sort_values(by="Ratings", ascending=False)

        # Select the top 5 places
        top_places = sorted_places.head(5)
        # print("TOP PLACES TO VISIT:")
        # print(top_places)


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


# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables before running the app
    app.run(debug=True)







