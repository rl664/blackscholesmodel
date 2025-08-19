import pandas as pd
import streamlit as st
import math
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib as plt
import plotly.express as px
import matplotlib.pyplot as plt


# Show the page title and description.
st.set_page_config(page_title="Black-Scholes Pricing Model", page_icon="📈")
st.title("📈Black-Scholes Pricing Model")
st.write(
    """
    Within the Black-Scholes framework, this interactive heatmap reveals how option values evolve across different spot price and volatility levels when the strike price is fixed. By visualising the pricing surface in this way, we gain direct insight into the sensitivity of options to underlying market movements and volatility shocks. 
 
    Click on the widgets below to explore!

    Made by Raphaelle Lassalle
    """
)


# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    df = pd.read_csv("data/movies_genres_summary.csv")
    return df


df = load_data()



# Show a slider widget with the years using `st.slider`. Defining the variables

min_vol = st.slider("Minimum Volatility for Heat Map", 0.01, 1.0, 0.1)
max_vol = st.slider("Maximum Volatility for Heat Map", 0.01, 1.0, 0.30)
min_spot = st.number_input("Minimum Spot Price", value = 80.0)
max_spot = st.number_input("Maximum Spot Price", value = 120.0)


# option parameters

current_price = st.number_input("Current Asset Price",value = 100.0)
strike_price = st.number_input("Strike Price", value = 100.0)
time_mat = st.number_input("Time to Maturity (Years)", value = 1.0)
volatility = st.number_input("Volatility (σ)", value = 0.2)
risk_free_rate= st.number_input("Risk-free Interest Rate", value = 0.05)


df_inputs = pd.DataFrame(
    [[current_price, strike_price, time_mat, volatility, risk_free_rate]],
    columns=[
        "Current Asset Price",
        "Strike Price",
        "Time to Maturity (Years)",
        "Volatility (σ)",
        "Risk-Free Interest Rate"
    ]
)

spot_prices = np.arange(min_spot, max_spot + 1, 1)   # step = 1
vols = np.arange(min_vol, max_vol + 0.01, 0.01)      # small step for volatility

# Display the data as a table using `st.dataframe`.
st.dataframe(
    df_inputs,
    use_container_width=True
)

# Display the data as an Altair chart using `st.altair_chart`.
df_chart = pd.melt(
    df_inputs, var_name="Parameter", value_name="Value"
)

d1 = (np.log(current_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_mat) / (volatility * np.sqrt(time_mat))
d2 = d1 - volatility * np.sqrt(time_mat)

call_price = current_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_mat) * norm.cdf(d2)
put_price = strike_price * np.exp(-risk_free_rate * time_mat) * norm.cdf(-d2) - current_price * norm.cdf(-d1)

st.write("Call Price:", round(call_price, 2))
st.write("Put Price:", round(put_price, 2))

heatmap_data = []

for S in spot_prices:
    for sigma in vols:
        d1 = (np.log(S / strike_price) + (risk_free_rate + 0.5 * sigma**2) * time_mat) / (sigma * np.sqrt(time_mat))
        d2 = d1 - sigma * np.sqrt(time_mat)
        call_price = S * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_mat) * norm.cdf(d2)
        put_price = strike_price * np.exp(-risk_free_rate * time_mat) * norm.cdf(-d2) - S * norm.cdf(-d1)
        heatmap_data.append({
            "Spot Price": S,
            "Volatility": sigma,
            "Call Price": call_price,
            "Put Price": put_price
        })
    
df_heatmap = pd.DataFrame(heatmap_data)

# Pivot the dataframe properly
pivot_call = df_heatmap.pivot(index="Volatility", columns="Spot Price", values="Call Price")

# Create the figure

step = 5  # show ticks and numbers every 5th value

# Create a masked annotation array
mask = np.full(pivot_call.shape, "", dtype=object)
for i in range(0, pivot_call.shape[0], step):
    for j in range(0, pivot_call.shape[1], step):
        mask[i, j] = f"{pivot_call.iloc[i, j]:.2f}"

# Make figure bigger depending on data size
fig, ax = plt.subplots(figsize=(len(pivot_call.columns)//2, len(pivot_call.index)//2))

sns.heatmap(
    pivot_call,
    cmap="RdYlGn",
    annot=mask,
    fmt="",
    xticklabels=False,
    yticklabels=False,
    square=True,  # make each cell square
    cbar_kws={"shrink": 0.7}  # optional: shrink colorbar
)

# Custom ticks every 5th value
ax.set_xticks(np.arange(0, len(pivot_call.columns), step))
ax.set_xticklabels(np.round(pivot_call.columns[::step], 2))

ax.set_yticks(np.arange(0, len(pivot_call.index), step))
ax.set_yticklabels(np.round(pivot_call.index[::step], 2))

ax.set_title("Call Option Prices Heatmap")
st.pyplot(fig)

# Pivot the dataframe for puts
pivot_put = df_heatmap.pivot(index="Volatility", columns="Spot Price", values="Put Price")

# Step for ticks and annotations
step = 5  

# Create a masked annotation array for puts
mask_put = np.full(pivot_put.shape, "", dtype=object)
for i in range(0, pivot_put.shape[0], step):
    for j in range(0, pivot_put.shape[1], step):
        mask_put[i, j] = f"{pivot_put.iloc[i, j]:.2f}"

# Make figure bigger depending on data size
fig, ax = plt.subplots(figsize=(len(pivot_put.columns)//2, len(pivot_put.index)//2))

sns.heatmap(
    pivot_put,
    cmap="RdYlGn",
    annot=mask_put,
    fmt="",
    xticklabels=False,
    yticklabels=False,
    square=True,
    cbar_kws={"shrink": 0.7}
)

# Custom ticks every 5th value
ax.set_xticks(np.arange(0, len(pivot_put.columns), step))
ax.set_xticklabels(np.round(pivot_put.columns[::step], 2))

ax.set_yticks(np.arange(0, len(pivot_put.index), step))
ax.set_yticklabels(np.round(pivot_put.index[::step], 2))

ax.set_title("Put Option Prices Heatmap")
st.pyplot(fig)

