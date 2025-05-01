import plotly.graph_objects as go #interactive graphs.
import pymysql #library used to connect to database.
from flask import Flask, request, jsonify #python web framework to create API's.
from flask_cors import CORS #(cross origin resource sharing) -> allowing the frontend to access the backend api's.
import pandas as pd#data manipulation
from statsmodels.tsa.statespace.sarimax import SARIMAX#time-series forcasting model.
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import numpy as np#numerical operations.
from flask_caching import Cache


#here i initilized a flask application and then allowed the app to accept requests from domains (frontend-backend)
app = Flask(__name__)
CORS(app)


#configuring my database with python
DB_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "thunder telenor site analysis"
    }



#returning the db connection and setting the query result returned as dictionaries rather than tuples.
def get_db_connection():
        return pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)



#funciton for fetching the ps data for a specific ts.
def fetch_power_data(timestamp):
        try:
            conn = get_db_connection()#func call.
            #object creation 'cursor' to interact with the db.
            cursor = conn.cursor()
            cursor.execute("SELECT mainkw, solarkw, dgkw, batterykw FROM thunder_data WHERE ts = %s", (timestamp,))#fetching the ps data for a specific ts.
            result = cursor.fetchone()#fetching only 1 result as for a specific ts there is only 1 result.
            conn.close()
            return result if result else None
        
        except pymysql.MySQLError as e: #if some error in fetching the data then the error will be logged.
            print("db error -> ", e)
            return None


#---------------------------------------------------LOGIN---------------------------------------------------
@app.route('/login', methods=['POST'])#api endpoint for post requests.
def login():
        '''
        this function fetches the username and password of the user from the db when the user tries to enter it in the web app.
        '''
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM user WHERE username = %s AND password = %s", (username, password))
            user = cursor.fetchone()
            conn.close()

            if user:
                return jsonify({"success": True, "message": "Login successful", "user_id": user["id"]})
            return jsonify({"success": False, "message": "Invalid credentials"}), 401

        except pymysql.MySQLError as e:
            print("Database Error:", e)
            return jsonify({"success": False, "message": "Database error"}), 500


#---------------------------------------------------GET LOAD SHARE(total power)-------------------------------------------------------
@app.route('/get-load-share', methods=['GET'])# API endpoint (/get-load-share) that listens for GET requests.
def get_load_share():
        '''
        in this function query is ran and ps data is extracted for a specific time stamp.
        and then out of all those extracted ps data it shows out of 100% how much each ps is used at a ts
        '''
        timestamp = request.args.get('ts')#get the ts from the url request.

        if not timestamp:
            return jsonify({"error": "Missing timestamp"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT mainkw, solarkw, dgkw, batterykw FROM thunder_data WHERE ts = %s"
        cursor.execute(query, (timestamp,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            return jsonify({"error": "No data found for the selected timestamp"}), 404

        mainkw, solarkw, dgkw, batterykw = result.values()
        total_power = mainkw + solarkw + dgkw + batterykw#calculating total power.

        if total_power == 0:#if total power 0 then give error for that ts
            return jsonify({"error": "current total power is zero at this ts, cannot calculate percentage"}), 400

        load_share ={
            "Main Grid": round((mainkw / total_power) * 100, 2),
            "Solar": round((solarkw / total_power) * 100, 2),
            "Diesel Generator": round((dgkw / total_power) * 100, 2),
            "Battery": round((batterykw / total_power) * 100, 2)
        }#finding % out of 100 upto 2 decimal places for each ps

        return jsonify(load_share)



#------------------------------------------------------------------TRAINING TIMESERIES MODEL---------------------------------------------------------
def fetch_loadshedding_data():
    """
    Fetches load shedding data for each timestamp and puts it in a sorted DataFrame.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT ts, ls_status FROM thunder_data WHERE ts >= '2025-01-01' AND ts < '2025-03-01' ORDER BY ts ASC"
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()

    # Convert to DataFrame and parse timestamps
    df = pd.DataFrame(data, columns=['ts', 'ls_status'])
    df['ts'] = pd.to_datetime(df['ts'], utc=True).dt.tz_convert('UTC')  # Ensure UTC timezone
    df = df.sort_values('ts')
    return df



def sarima_model_training():
    """
    Trains a SARIMA model on load shedding data and generates predictions for February 2025.
    """
    # Fetch and preprocess data
    df = fetch_loadshedding_data()
    df.set_index('ts', inplace=True)

    # Resample to daily level using mean to smooth fluctuations
    df = df.resample('D').mean().fillna(0)
    df['ls_percentage'] = df['ls_status'] * 100  # Convert to percentage

    # Actual data for February 2025
    feb_actual = df.loc["2025-02-01":"2025-02-18"].reset_index()

    # Define SARIMA parameters
    order = (2, 1, 2)  # (p, d, q)
    seasonal_order = (1, 1, 1, 7)  # (P, D, Q, S)

    # Train SARIMA model
    model = SARIMAX(df['ls_status'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Generate predictions for February 2025
    predict_dates = pd.date_range(start="2025-02-01", end="2025-02-18", freq="D")
    prediction = model_fit.predict(start=len(df), end=len(df) + len(predict_dates) - 1)

    # Create DataFrame for predictions
    predicted_df = pd.DataFrame({'ts': predict_dates, 'predicted_ls_status': prediction.values})
    predicted_df['ts'] = pd.to_datetime(predicted_df['ts'], utc=True).dt.tz_convert('UTC')  # Ensure UTC timezone
    predicted_df['predicted_ls_percentage'] = predicted_df['predicted_ls_status'] * 100

    # Combine actual and predicted data using pd.concat
    result_df = pd.concat([feb_actual, predicted_df], axis=1)
    result_df.fillna(0, inplace=True)  # Fill missing values with 0
    return result_df



cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route("/loadshedding/prediction", methods=["GET"])
@cache.cached(timeout=3600)  # Cache for 1 hour
def get_loadshedding_prediction():
    result_df = sarima_model_training()
    data_json = result_df.to_dict(orient="records")  # Convert DataFrame to JSON-compatible format
    return jsonify(data_json)



@app.route("/loadshedding", methods=["GET"])
def get_loadshedding():
    """
    Returns actual load shedding data as JSON for frontend.
    """
    # Fetch and preprocess data
    df = fetch_loadshedding_data()
    df.set_index('ts', inplace=True)

    # Resample to daily level using mean to smooth fluctuations
    df = df.resample('D').mean().fillna(0)
    df['ls_percentage'] = df['ls_status'] * 100  # Convert to percentage

    # Convert to JSON-compatible format
    data_json = df.reset_index().to_dict(orient="records")
    return jsonify(data_json)




#-------------------------------------------------------GET TIME STAMP------------------------------------------------------
@app.route('/get-timestamps', methods=['GET'])
def get_timestamps():
    """
    Fetches all timestamps for 2025 and returns them as a JSON response.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT ts FROM thunder_data WHERE ts >= '2025-01-01' AND ts < '2025-03-01' ORDER BY ts DESC LIMIT 8596"
        cursor.execute(query)
        timestamps = [row['ts'].strftime('%Y-%m-%d %H:%M:%S') for row in cursor.fetchall()]
        conn.close()
        return jsonify(timestamps)
    except pymysql.MySQLError as e:
        print("Database Error:", e)
        return jsonify({"error": "Failed to fetch timestamps"}), 500



'''defining an api endpoint '/fetch-chart' that listens to the 'GET' requests. 
so basically url will send a request here to fetch a chart and using flask and CORS we make that work.'''



#----------------------------------------------------FETCH CHART-------------------------------------------------------
@app.route('/fetch-chart', methods=['GET']) 
def fetch_chart():
        timestamp = request.args.get('ts')
#when url request arrives it gives a ts with it so .get is used to retrieve that ts from the url so the ts can be used to plot 
        
        if not timestamp:
            return jsonify({"error": "Timestamp is required"}), 400#if no ts then error

        data = fetch_power_data(timestamp)#calling the fetch ps function to retrieve the data extracted from the query.

        if data:
            '''
            so basically in the dataset, new ts with values arrive every 5 minutes so the dataset is quite large and complex,
            with no assurance that there is no NULL values because they may be, so to ensure that the chart is plotted properly
            if the result from db is missing or none it is defaulted to 0 for all the ps(mainkw, solarkw,dgkw,batterykw).
            '''
            mainkw = data.get('mainkw', 0) or 0
            solarkw = data.get('solarkw', 0) or 0
            dgkw = data.get('dgkw', 0) or 0
            batterykw = data.get('batterykw', 0) or 0

            sources = ['Main', 'Solar', 'DG', 'Battery']#list
            values = [mainkw, solarkw, dgkw, batterykw]

            fig = go.Figure()
            fig.add_trace(go.Bar(#barchart plot.
                x=sources,
                y=values,
                marker_color=['blue', 'yellow', 'red', 'green']
            ))

            fig.update_layout(
                title=f"Power Source Distribution for {timestamp}",
                xaxis_title="Power Sources",
                yaxis_title="Power (kW)",
                plot_bgcolor="#161b22",
                paper_bgcolor="#161b22",
                font=dict(color="white")
            )

            '''
            converting the barchar plot to json, the reason is that when i plot this in my front end it wont be interactive so make it work,
            ive to convert it into json so the frontend can render it as an interactive chart.
            '''
            graph_json = fig.to_json()

            return jsonify({#sending the json data and chart to front end.
                "graph": graph_json,
                "mainkw": mainkw,
                "solarkw": solarkw,
                "dgkw": dgkw,
                "batterykw": batterykw
            })

        else:
            return jsonify({"error": "No data found for the provided timestamp"}), 404







#--------------------------------------------------------DAILY LOAD SHEDDING---------------------------------------------------------
@app.route("/daily-load-shedding", methods=["GET"])
def get_daily_load_shedding():
    """
    Calculates daily LS percentage and total hours of load shedding based on available entries per day.
    """
    try:
        # Fetch and preprocess data
        df = fetch_loadshedding_data()
        df['ts'] = pd.to_datetime(df['ts'])
        df['date'] = df['ts'].dt.date

        # Total entries per day
        total_per_day = df.groupby('date').size().reset_index(name='total')

        # LS entries per day
        ls_per_day = df[df['ls_status'] == 1].groupby('date').size().reset_index(name='ls_count')

        # Merge and calculate percentage
        merged = pd.merge(total_per_day, ls_per_day, on='date', how='left')
        merged['ls_count'] = merged['ls_count'].fillna(0)
        merged['ls_percentage'] = round((merged['ls_count'] / merged['total']) * 100, 2)

        # Calculate total hours of load shedding
        interval_minutes = 5  # Assuming each entry corresponds to a 5-minute interval
        merged['ls_hours'] = round((merged['ls_count'] * interval_minutes) / 60, 2)

        # Return results as JSON
        return jsonify(merged.rename(columns={"date": "ts"}).to_dict(orient="records"))

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Internal server error"}), 500



@app.route('/battery-analysis', methods=['GET'])
def battery_analysis():
    """
    Analyze battery charging/discharging behavior for all timestamps in the database.
    """
    try:
        # Establish a database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query to fetch all data from the database
        query = """
        SELECT sitecode, ts, main_contrib_batt_kwh, dg_contrib_batt_kwh, solar_contrib_batt_kwh,
               csu_ana_batt_total_curr, ac_dig_ac_source, source_tag, ls_status
        FROM thunder_data 
        WHERE ts >= '2025-01-01' AND ts < '2025-03-01' 
        ORDER BY ts
        """
        cursor.execute(query)
        data = cursor.fetchall()
        conn.close()

        # Check if data exists
        if not data:
            return jsonify({"error": "No data found in the database"}), 404

        # Process the fetched data
        results = []
        for row in data:
            try:
                # Format the timestamp properly
                timestamp = row['ts'].strftime('%Y-%m-%d %H:%M:%S')
            except AttributeError:
                # Handle invalid or missing timestamps gracefully
                timestamp = "Invalid Timestamp"

            # Determine charging/discharging action
            action = "Discharging" if row['csu_ana_batt_total_curr'] > 0 else "Charging"

            # Determine the power source
            source = "External Source" if row['ac_dig_ac_source'] == 1 else "Battery"

            # Calculate contributions
            contributions = {
                "Main": row['main_contrib_batt_kwh'],
                "Diesel Generator": row['dg_contrib_batt_kwh'],
                "Solar": row['solar_contrib_batt_kwh']
            }

            # Determine energy flow direction
            energy_flow = {
                "Main": "Energy Taken" if row['main_contrib_batt_kwh'] < 0 else "Energy Added",
                "Diesel Generator": "Energy Taken" if row['dg_contrib_batt_kwh'] < 0 else "Energy Added",
                "Solar": "Energy Taken" if row['solar_contrib_batt_kwh'] < 0 else "Energy Added"
            }

            # Append the result for this row
            results.append({
                "timestamp": timestamp,
                "action": action,
                "source": source,
                "contributions": contributions,
                "energy_flow": energy_flow,
                "load_shedding": "True" if row['ls_status'] else "False"
            })

        # Return the results as JSON
        return jsonify(results)

    except pymysql.MySQLError as e:
        # Log and handle database errors
        print("Database Error:", e)
        return jsonify({"error": "Database error"}), 500

 
if __name__ == '__main__':
        app.run(host="0.0.0.0", port=5000, debug=True)
