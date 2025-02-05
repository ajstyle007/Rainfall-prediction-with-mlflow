import streamlit as st
import numpy as np
import pandas as pd
import pickle
import mlflow
import dagshub

# Load trained model
pipe = pickle.load(open("model.pkl", "rb"))

# Load dataset (assuming it contains original min/max values)
df = pd.read_csv("rainfall_newdf.csv")

# features = ['pressure', 'temparature', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# st.title("RainFall Prediction using Machine Learning")
st.markdown(
        "<h1 style='text-align: center; color: #1a5276; margin-top: -50px; font-weight: bold; font-size: 33px;'>RainFall 🌦️ Prediction using Machine Learning</h1>", 
        unsafe_allow_html=True
    )

tab1, tab2, tab3 = st.tabs(["Overview", "Prediction", "Experiment Tracking"])

with tab1:

    col1, col2 = st.columns(2)

    col1.image("rain.png", width= 300)
    col2.image("weather.png", width= 300)

    st.write("Rain is essential for life, as it replenishes water sources, supports agriculture, and maintains ecosystems. However, due to climate change, deforestation, and urbanization, rainfall patterns are becoming unpredictable, leading to droughts or floods. Here’s how we can save rain and use it efficiently:")
    
    st.subheader("1. Rainfall Prediction & Early Planning")
    st.markdown("- **Weather Forecasting**: Using advanced weather models, AI, and satellite data, we can predict when and how much rain will fall.")
    st.markdown("- **Smart Water Storage**: If heavy rain is predicted, we can prepare reservoirs, lakes, and rainwater harvesting tanks to capture and store the excess water.")
    st.markdown("- **Flood Control Measures**: Accurate predictions allow authorities to take precautions like opening dam gates in a controlled manner to prevent floods while conserving water.")
    st.markdown("- **Efficient Irrigation**: Farmers can plan irrigation schedules based on predicted rainfall, reducing unnecessary water usage.")
    st.markdown("- **Disaster Preparedness**: Governments and communities can take proactive steps to protect infrastructure, crops, and people from extreme rainfall events.")

    st.subheader("2. Rainwater Harvesting")
    st.markdown("- Install rainwater collection systems on rooftops to store rain for household use.")
    st.markdown("- Use rain barrels to collect rainwater for gardening and cleaning.")
    st.markdown("- Build check dams and percolation pits to recharge groundwater.")

    st.subheader("3. Afforestation & Reforestation")
    st.markdown("- Plant trees to increase rainfall retention and prevent soil erosion.")
    st.markdown("- Protect forests, as they help in maintaining local weather patterns.")

    st.subheader("4. Reducing Pollution")
    st.markdown("- Prevent industrial waste and sewage from contaminating water bodies.")
    st.markdown("- Avoid excessive use of pesticides and fertilizers that can pollute rainwater.")

    st.subheader("5. Sustainable Urban Planning")
    st.markdown("- Design cities with green roofs and permeable pavements to absorb rainwater.")
    st.markdown("- Restore wetlands, which act as natural sponges to store excess rainwater.")

    st.subheader("6. Preventing Water Wastage")
    st.markdown("- Fix leaks and use water-efficient appliances.")
    st.markdown("- Practice rain-fed agriculture to reduce dependence on groundwater.")

    st.write("Using rainfall prediction alongside conservation techniques like rainwater harvesting, afforestation, and urban planning, we can make the most of every drop of rain. 🌧️💧")




with tab2:
    # Streamlit UI
    st.markdown(
        "<h1 style='text-align: center; color: green; margin-top: -30px; font-weight: bold; font-size: 45px;'>Rainfall Prediction</h1>", 
        unsafe_allow_html=True
    )
    st.write("")

    # st.sidebar.title("RainFall")

    col1, col2, col3 = st.columns(3)

    with col1:
        pressure = st.number_input("Pressure (hPa)", min_value=0.0, max_value=2000.0, value=1013.0)
        cloud = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=70.0)
        windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=70.0, value=45.0)

    with col2:
        temparature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        sunshine = st.number_input("Sunshine Hours", min_value=0.0, max_value=20.0, value=8.0)

    with col3:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
        winddirection = st.number_input("Wind Direction (°)", min_value=0.0, max_value=300.0, value=180.0)

    # Prepare input data
    input_df = (pressure, temparature, humidity, cloud, sunshine, winddirection, windspeed)
    input_df = pd.DataFrame([input_df], ['pressure', 'temparature', 'humidity', 'cloud', 'sunshine','winddirection', 'windspeed'])

    if st.button("Make Prediction"):
        prediction = pipe.predict(input_df)
        # st.write("Prediction Result: ", "**Rainfall**" if prediction[0] == 1 else "**No Rainfall**")

        if prediction[0] == 1:
            st.markdown("<h3 style='color: red; text-align: center;'>Prediction: Rainfall 🌧️</h3>", unsafe_allow_html=True)
            st.markdown("<div style='color: cyan; text-align: center;'>The model predicts that there will be rainfall based on current conditions.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green; text-align: center;'>Prediction: No Rainfall ☀️</h3>", unsafe_allow_html=True)
            st.markdown("<div style='color: cyan; text-align: center;'>The model predicts no rainfall based on the input parameters.</div>", unsafe_allow_html=True)

with tab3:

    dagshub.auth.add_app_token("59e05d02bd7485a2152aec154bdfd08c78b6cf48")
    dagshub.init(repo_owner='ajstyle007', repo_name='my-first-repo', mlflow=True)


    mlflow.set_experiment("Rainfall")
    mlflow.set_tracking_uri(uri="https://dagshub.com/ajstyle007/my-first-repo.mlflow")
    run_id = "2e8e5c1f605a454c832420d30b9b4c42"  # Replace with a specific Run ID

    all_runs = mlflow.search_runs(search_all_experiments=True)
    # st.write(all_runs)
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    parameters = client.get_run(run_id).data.params


    mlflow_url = "https://dagshub.com/ajstyle007/my-first-repo.mlflow"

    st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <p style="font-size: 15px; color: #32CD32; margin: 0;">Click the button to access MLflow for comprehensive experiment tracking. ►</p>
        <a href="{mlflow_url}" target="_blank">
            <button style="
                background-color: black; color: white; padding: 10px 20px; border-radius: 5px; font-size: 16px; border: none; cursor: pointer;
            ">
                Go to MLflow UI ↗️
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True)

    st.write("")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", np.round((metrics)["Accuracy"], 3))
        col2.metric("Precision_0", np.round((metrics)["Precision_0"], 3))
        col3.metric("Recall_0", np.round((metrics)["Recall_0"], 3))

        col1.metric("Precision_1", np.round((metrics)["Precision_1"], 3))
        col2.metric("Recall_1", np.round((metrics)["Recall_1"], 3))
        col3.metric("f1_score_macro", np.round((metrics)["f1_score_macro"], 3))

    
    # st.write(pd.DataFrame(parameters, columns=["Parameters", "Values"]))
    # Convert dictionary to DataFrame
    df_params = pd.DataFrame(list(parameters.items()), columns=["Parameter", "Value"])

    # Display as a table
    st.table(df_params)


    



