import streamlit as st
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Inisialisasi data
if 'user_data' not in st.session_state:
    original_data = {
        "Coffee Beans": {"unit": "kg", "demand": 78, "ordering_cost": 30000, "holding_cost": 4361.48, "pre_eoq_q": 2.108, "pre_eoq_f": 37},
        "Fresh Milk": {"unit": "liter", "demand": 576, "ordering_cost": 20000, "holding_cost": 6256.37, "pre_eoq_q": 24, "pre_eoq_f": 24},
        "Sugar": {"unit": "kg", "demand": 47, "ordering_cost": 5000, "holding_cost": 6163.84, "pre_eoq_q": 1, "pre_eoq_f": 47},
        "Vanilla Syrup": {"unit": "bottle", "demand": 24, "ordering_cost": 6667, "holding_cost": 3079.43, "pre_eoq_q": 1, "pre_eoq_f": 24},
        "Hazelnut Syrup": {"unit": "bottle", "demand": 24, "ordering_cost": 6667, "holding_cost": 3079.43, "pre_eoq_q": 1, "pre_eoq_f": 24},
        "Caramel Syrup": {"unit": "bottle", "demand": 24, "ordering_cost": 6667, "holding_cost": 3079.43, "pre_eoq_q": 1, "pre_eoq_f": 24}
    }
    st.session_state.user_data = {material: data.copy() for material, data in original_data.items()}

if 'lead_times' not in st.session_state:
    st.session_state.lead_times = {
        "Coffee Beans": 3, "Fresh Milk": 2, "Sugar": 1,
        "Vanilla Syrup": 2, "Hazelnut Syrup": 2, "Caramel Syrup": 2
    }

if 'safety_stocks' not in st.session_state:
    st.session_state.safety_stocks = {
        "Coffee Beans": 0.70, "Fresh Milk": 3.45, "Sugar": 0.16,
        "Vanilla Syrup": 0.16, "Hazelnut Syrup": 0.16, "Caramel Syrup": 0.16
    }

if 'conventional_tics' not in st.session_state:
    st.session_state.conventional_tics = {
        "Coffee Beans": 1114597.24, "Fresh Milk": 555076.44, "Sugar": 238081.92,
        "Vanilla Syrup": 161539.72, "Hazelnut Syrup": 161539.72, "Caramel Syrup": 161539.72
    }

# Fungsi utilitas
def darken_color(hex_color, factor):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    darker_rgb = tuple(max(0, c - factor) for c in rgb)
    return '#%02x%02x%02x' % darker_rgb

def calculate_eoq(D, OC, CC):
    if CC <= 0:
        return 0
    return math.sqrt((2 * D * OC) / CC)

def calculate_metrics(material, period):
    data = st.session_state.user_data[material]
    D = data["demand"]
    OC = data["ordering_cost"]
    CC = data["holding_cost"]
    LT = st.session_state.lead_times[material]
    SS = st.session_state.safety_stocks[material]
    unit = data["unit"]

    EOQ = calculate_eoq(D, OC, CC)
    F = D / EOQ if EOQ > 0 else 0
    TOC = (D / EOQ) * OC if EOQ > 0 else 0
    TCC = (EOQ / 2) * CC
    TIC = TOC + TCC

    daily_usage = D / period
    ROP = (daily_usage * LT) + SS

    average_inventory = EOQ / 2
    merchandise_turnover = D / average_inventory if average_inventory > 0 else 0

    return {
        "Material": material,
        "Unit": unit,
        "EOQ": EOQ,
        "Orders/Year": F,
        "TOC": TOC,
        "TCC": TCC,
        "TIC": TIC,
        "ROP": ROP,
        "Avg Inventory": average_inventory,
        "Turnover": merchandise_turnover
    }

def calculate_all_materials(period):
    results = []
    for material in st.session_state.user_data:
        metrics = calculate_metrics(material, period)
        results.append(metrics)
    return results

def plot_eoq_graph(material, D, OC, CC, EOQ_val, TIC_val):
    fig, ax = plt.subplots(figsize=(10, 6))

    if CC == 0 or D == 0:
        ax.text(0.5, 0.5, "Cannot plot: Holding Cost or Demand is zero",
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12, color='red')
        return fig

    q_min = max(1, int(EOQ_val * 0.1))
    q_max = int(EOQ_val * 2.5) + 50

    num_points = 200
    q_vals = np.linspace(q_min, q_max, num_points)

    toc_vals = (D / q_vals) * OC
    tcc_vals = (q_vals / 2) * CC
    tc_vals = toc_vals + tcc_vals

    ax.plot(q_vals, tc_vals, label='Total Inventory Cost (TIC)', color='#4CAF50', linewidth=2.5, alpha=0.9)
    ax.plot(q_vals, toc_vals, label='Total Ordering Cost (TOC)', color='#1E88E5', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.plot(q_vals, tcc_vals, label='Total Holding Cost (TCC)', color='#E91E63', linestyle=':', linewidth=1.5, alpha=0.8)

    ax.axvline(x=EOQ_val, color='gray', linestyle='-.', label=f'EOQ = {EOQ_val:,.2f}', alpha=0.7)
    ax.axhline(y=TIC_val, color='gray', linestyle=':', label=f'Min TIC = Rp {TIC_val:,.2f}', alpha=0.7)
    ax.plot(EOQ_val, TIC_val, 'o', color='#FF5722', markersize=9, label='Optimal Point', markeredgecolor='white', markeredgewidth=1.5)

    ax.set_title(f"EOQ Cost Curve for {material}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Order Quantity (Q)")
    ax.set_ylabel("Cost (Rp)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)

    return fig

# UI Streamlit
st.set_page_config(
    page_title="Chopfee Coffee Shop - EOQ & ROP Analyzer",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
<style>
    .stApp {
        background-color: #333333; /* Darker background color */
        color: #FFFFFF; /* White text for better contrast */
    }
    .header-text {
        color: #ffffff;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subheader {
        color: #87CEEB; /* Lighter blue for subheaders on dark background */
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .card {
        background-color: #444444; /* Darker card background */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); /* Adjust shadow for dark background */
        color: #FFFFFF; /* White text inside cards */
    }
    .highlight {
        background-color: #2F4F4F; /* Darker highlight */
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        color: #FFFFFF; /* White text for highlight */
    }
    .metric-box {
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        margin: 10px 0;
        background-color: #555555; /* Darker metric box background */
        color: #FFFFFF; /* White text inside metric box */
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2); /* Adjust shadow for dark background */
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold;
        padding: 10px 20px !important;
        border-radius: 5px !important;
    }
    .stButton>button:hover {
        background-color: #388E3C !important;
    }
    /* Adjust text color for general markdown elements if needed */
    body {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-text">☕ Chopfee Coffee Shop - EOQ & ROP Analyzer</div>', unsafe_allow_html=True)

# Tab utama
tabs = st.tabs([
    "📋 Input Parameters",
    "📊 EOQ & ROP Results",
    "💰 Cost Comparison",
    "📈 EOQ Graph Visualization",
    "📑 All Materials Summary"
])

# Tab 1: Input Parameters
with tabs[0]:
    st.subheader("General Settings")
    period = st.number_input("Analysis Period (days)", min_value=1, value=334, step=1)

    st.divider()
    st.subheader("Material Parameters")

    col1, col2 = st.columns(2)
    selected_material = col1.selectbox("Select Raw Material", list(st.session_state.user_data.keys()))

    material_data = st.session_state.user_data[selected_material]
    lead_time = st.session_state.lead_times[selected_material]
    safety_stock = st.session_state.safety_stocks[selected_material]

    with st.form("material_form"):
        col1, col2, col3 = st.columns(3)

        demand = col1.number_input(
            "Annual Demand (D)",
            min_value=0.0,
            value=float(material_data["demand"]),
            step=1.0
        )

        ordering_cost = col2.number_input(
            "Ordering Cost (S) (Rp)",
            min_value=0.0,
            value=float(material_data["ordering_cost"]),
            step=1000.0
        )

        holding_cost = col3.number_input(
            "Holding Cost (H) (Rp/unit)",
            min_value=0.0,
            value=float(material_data["holding_cost"]),
            step=100.0
        )

        col1, col2 = st.columns(2)
        lead_time = col1.number_input(
            "Lead Time (L) (days)",
            min_value=0,
            value=int(lead_time),
            step=1
        )

        safety_stock = col2.number_input(
            "Safety Stock (SS) (units)",
            min_value=0.0,
            value=float(safety_stock),
            step=0.1
        )

        submitted = st.form_submit_button("Calculate & Update All Materials")
        reset = st.form_submit_button("Reset Fields")

        if submitted:
            st.session_state.user_data[selected_material]["demand"] = demand
            st.session_state.user_data[selected_material]["ordering_cost"] = ordering_cost
            st.session_state.user_data[selected_material]["holding_cost"] = holding_cost
            st.session_state.lead_times[selected_material] = lead_time
            st.session_state.safety_stocks[selected_material] = safety_stock
            st.success("Parameters updated successfully!")

        if reset:
            original_data = {
                "Coffee Beans": {"unit": "kg", "demand": 78, "ordering_cost": 30000, "holding_cost": 4361.48, "pre_eoq_q": 2.108, "pre_eoq_f": 37},
                "Fresh Milk": {"unit": "liter", "demand": 576, "ordering_cost": 20000, "holding_cost": 6256.37, "pre_eoq_q": 24, "pre_eoq_f": 24},
                "Sugar": {"unit": "kg", "demand": 47, "ordering_cost": 5000, "holding_cost": 6163.84, "pre_eoq_q": 1, "pre_eoq_f": 47},
                "Vanilla Syrup": {"unit": "bottle", "demand": 24, "ordering_cost": 6667, "holding_cost": 3079.43, "pre_eoq_q": 1, "pre_eoq_f": 24},
                "Hazelnut Syrup": {"unit": "bottle", "demand": 24, "ordering_cost": 6667, "holding_cost": 3079.43, "pre_eoq_q": 1, "pre_eoq_f": 24},
                "Caramel Syrup": {"unit": "bottle", "demand": 24, "ordering_cost": 6667, "holding_cost": 3079.43, "pre_eoq_q": 1, "pre_eoq_f": 24}
            }
            st.session_state.user_data = {material: data.copy() for material, data in original_data.items()}
            st.session_state.lead_times = {
                "Coffee Beans": 3, "Fresh Milk": 2, "Sugar": 1,
                "Vanilla Syrup": 2, "Hazelnut Syrup": 2, "Caramel Syrup": 2
            }
            st.session_state.safety_stocks = {
                "Coffee Beans": 0.70, "Fresh Milk": 3.45, "Sugar": 0.16,
                "Vanilla Syrup": 0.16, "Hazelnut Syrup": 0.16, "Caramel Syrup": 0.16
            }
            st.success("All fields reset to default values!")

# Tab 2: EOQ & ROP Results
with tabs[1]:
    if not selected_material:
        st.warning("Please select a material in the Input Parameters tab")
    else:
        st.subheader(f"Optimal Inventory Metrics for {selected_material}")

        # Hitung metrik
        data = st.session_state.user_data[selected_material]
        D = data["demand"]
        OC = data["ordering_cost"]
        CC = data["holding_cost"]
        LT = st.session_state.lead_times[selected_material]
        SS = st.session_state.safety_stocks[selected_material]
        unit = data["unit"]

        EOQ = calculate_eoq(D, OC, CC)
        F = D / EOQ if EOQ > 0 else 0
        TOC = (D / EOQ) * OC if EOQ > 0 else 0
        TCC = (EOQ / 2) * CC
        TIC = TOC + TCC

        daily_usage = D / period
        ROP = (daily_usage * LT) + SS
        reorder_time = period / F if F > 0 else 0

        average_inventory = EOQ / 2
        merchandise_turnover = D / average_inventory if average_inventory > 0 else 0

        # Tampilkan hasil dalam kolom
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Basic Parameters")
            st.markdown(f"**Annual Demand (D):** {D:,.2f} {unit}/year")
            st.markdown(f"**Ordering Cost per Order (S):** Rp {OC:,.2f}")
            st.markdown(f"**Holding Cost per Unit (H):** Rp {CC:,.2f}/{unit}/year")

            st.markdown("### EOQ Metrics")
            st.markdown(f"**Optimal Order Quantity (EOQ):** {EOQ:,.2f} {unit}")
            st.markdown(f"**Number of Orders per Year (F):** {F:,.2f} orders")
            st.markdown(f"**Total Annual Ordering Cost (TOC):** Rp {TOC:,.2f}")
            st.markdown(f"**Total Annual Holding Cost (TCC):** Rp {TCC:,.2f}")
            st.markdown(f"<div class='highlight'>Total Annual Inventory Cost (TIC): Rp {TIC:,.2f}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### Reorder Point Metrics")
            st.markdown(f"**Lead Time (L):** {LT} days")
            st.markdown(f"**Safety Stock (SS):** {SS} {unit}")
            st.markdown(f"**Daily Usage:** {daily_usage:,.2f} {unit}/day")
            st.markdown(f"**Reorder Point (ROP):** {ROP:,.2f} {unit}")
            st.markdown(f"**Reorder Time (days between orders):** {reorder_time:,.2f} days")

            st.markdown("### Inventory Turnover Metrics")
            st.markdown(f"**Average Inventory:** {average_inventory:,.2f} {unit}")
            st.markdown(f"<div class='highlight'>Merchandise Turnover: {merchandise_turnover:,.2f} times/year</div>", unsafe_allow_html=True)

        # Penjelasan
        st.divider()
        current_pre_eoq_cost = st.session_state.conventional_tics.get(selected_material)
        if current_pre_eoq_cost is not None:
            explanation = f"""
            Untuk **{selected_material}**, kuantitas pesanan optimal (EOQ) adalah **{EOQ:,.2f} {unit}** per pesanan.
            Ini berarti Chopfee Coffee Shop harus memesan **{EOQ:,.0f} {unit}** {selected_material} setiap kali mereka melakukan pemesanan.
            Dengan strategi ini, mereka akan melakukan **{F:,.1f}** pesanan dalam setahun,
            dengan jarak antar pesanan sekitar **{reorder_time:,.1f} hari**.

            Titik pemesanan kembali (ROP) untuk {selected_material} adalah **{ROP:,.2f} {unit}**.
            Ini berarti ketika persediaan {selected_material} mencapai **{ROP:,.0f} {unit}**, pesanan baru harus segera ditempatkan
            untuk memastikan ketersediaan selama waktu tunggu pengiriman ({LT} hari),
            ditambah stok pengaman **{SS} {unit}** untuk mengantisipasi fluktuasi permintaan atau keterlambatan pengiriman.

            **Merchandise Turnover** (Perputaran Persediaan) untuk {selected_material} adalah sekitar **{merchandise_turnover:,.2f} kali per tahun**,
            menunjukkan seberapa efisien persediaan dijual atau digunakan dalam periode tersebut.
            Penerapan model EOQ ini diperkirakan dapat mengurangi total biaya inventaris tahunan dari
            **Rp {current_pre_eoq_cost:,.2f}** (metode konvensional) menjadi **Rp {TIC:,.2f}** (metode EOQ).
            """
        else:
            explanation = f"""
            Untuk **{selected_material}**, EOQ adalah **{EOQ:,.2f} {unit}** dan ROP adalah **{ROP:,.2f} {unit}**.
            Dengan {F:,.1f} pesanan per tahun, dan pemesanan kembali setiap {reorder_time:,.1f} hari,
            total biaya inventaris tahunan dengan EOQ adalah **Rp {TIC:,.2f}**.

            **Merchandise Turnover** (Perputaran Persediaan) untuk {selected_material} adalah sekitar **{merchandise_turnover:,.2f} kali per tahun**.
            """

        st.markdown(explanation)

# Tab 3: Cost Comparison
with tabs[2]:
    st.subheader("Total Inventory Cost Comparison (Conventional vs. EOQ)")

    comparison_data = []
    total_before = 0
    total_after = 0

    for material in st.session_state.user_data:
        metrics = calculate_metrics(material, period)
        before_cost = st.session_state.conventional_tics.get(material, 0)
        after_cost = metrics["TIC"]
        savings = before_cost - after_cost
        reduction_percent = (savings / before_cost) * 100 if before_cost != 0 else 0

        total_before += before_cost
        total_after += after_cost

        comparison_data.append({
            "Raw Material": material,
            "Conventional Cost (Rp)": before_cost,
            "EOQ Cost (Rp)": after_cost,
            "Savings (Rp)": savings,
            "Reduction (%)": reduction_percent
        })

    total_savings = total_before - total_after
    total_reduction_percent = (total_savings / total_before) * 100

    comparison_data.append({
        "Raw Material": "TOTAL",
        "Conventional Cost (Rp)": total_before,
        "EOQ Cost (Rp)": total_after,
        "Savings (Rp)": total_savings,
        "Reduction (%)": total_reduction_percent
    })

    # Format dataframe untuk tampilan
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison["Conventional Cost (Rp)"] = df_comparison["Conventional Cost (Rp)"].apply(lambda x: f"Rp {x:,.2f}")
    df_comparison["EOQ Cost (Rp)"] = df_comparison["EOQ Cost (Rp)"].apply(lambda x: f"Rp {x:,.2f}")
    df_comparison["Savings (Rp)"] = df_comparison["Savings (Rp)"].apply(lambda x: f"Rp {x:,.2f}")
    df_comparison["Reduction (%)"] = df_comparison["Reduction (%)"].apply(lambda x: f"{x:,.1f}%")

    # Tampilkan tabel
    st.dataframe(df_comparison, height=600)

    # Ringkasan
    st.divider()
    summary_text = f"""
    Berdasarkan analisis EOQ, Chopfee Coffee Shop berpotensi menghemat total biaya inventaris tahunan sebesar
    **Rp {total_savings:,.2f}**, yang merepresentasikan penurunan biaya sebesar **{total_reduction_percent:.1f}%**
    dari total biaya konvensional Rp {total_before:,.2f} menjadi Rp {total_after:,.2f}.
    Ini konsisten dengan temuan studi yang menunjukkan potensi penghematan signifikan melalui optimasi inventaris.
    """
    st.markdown(summary_text)

# Tab 4: EOQ Graph Visualization
with tabs[3]:
    if not selected_material:
        st.warning("Please select a material in the Input Parameters tab")
    else:
        st.subheader(f"EOQ Cost Curve Visualization for {selected_material}")

        # Hitung metrik
        data = st.session_state.user_data[selected_material]
        D = data["demand"]
        OC = data["ordering_cost"]
        CC = data["holding_cost"]

        EOQ = calculate_eoq(D, OC, CC)
        TOC = (D / EOQ) * OC if EOQ > 0 else 0
        TCC = (EOQ / 2) * CC
        TIC = TOC + TCC

        # Generate plot
        fig = plot_eoq_graph(selected_material, D, OC, CC, EOQ, TIC)
        st.pyplot(fig)

        # Ringkasan grafik
        st.divider()
        summary_text = f"""
        **Ringkasan Grafik EOQ untuk {selected_material}:**

        Titik optimal (EOQ) pada grafik ini menunjukkan kuantitas pesanan yang meminimalkan Total Biaya Inventaris (TIC).

        - **EOQ optimal:** {EOQ:,.2f} unit
        - **Biaya Inventaris Tahunan Minimal (TIC):** Rp {TIC:,.2f}

        Pada titik ini, Total Biaya Pemesanan (TOC) dan Total Biaya Penyimpanan (TCC) adalah seimbang,
        menunjukkan efisiensi maksimum dalam pengelolaan inventaris.
        """
        st.markdown(summary_text)

# Tab 5: All Materials Summary
with tabs[4]:
    st.subheader("EOQ & ROP Results for All Raw Materials")
    st.caption("Tabel ini menunjukkan metrik inventaris optimal yang dihitung menggunakan model EOQ untuk semua bahan baku.")

    # Hitung semua metrik
    all_metrics = calculate_all_materials(period)

    # Format dataframe untuk tampilan
    df_all = pd.DataFrame(all_metrics)
    df_all["TOC"] = df_all["TOC"].apply(lambda x: f"Rp {x:,.2f}")
    df_all["TCC"] = df_all["TCC"].apply(lambda x: f"Rp {x:,.2f}")
    df_all["TIC"] = df_all["TIC"].apply(lambda x: f"Rp {x:,.2f}")

    # Format kolom numerik
    numeric_cols = ["EOQ", "Orders/Year", "ROP", "Avg Inventory", "Turnover"]
    for col in numeric_cols:
        df_all[col] = df_all[col].apply(lambda x: f"{x:,.2f}")

    # Tampilkan tabel
    st.dataframe(df_all, height=600)

    # Tombol ekspor
    st.divider()
    col1, col2 = st.columns(2)

    # Ekspor sebagai CSV
    csv = df_all.to_csv(index=False).encode('utf-8')
    col1.download_button(
        label="Download as CSV",
        data=csv,
        file_name="chopfee_eoq_summary.csv",
        mime="text/csv"
    )

    # Ekspor sebagai Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_all.to_excel(writer, index=False, sheet_name='EOQ Summary')
    excel_data = output.getvalue()

    col2.download_button(
        label="Download as Excel",
        data=excel_data,
        file_name="chopfee_eoq_summary.xlsx",
        mime="application/vnd.ms-excel"
    )

# Footer
st.divider()
st.caption("© 2023 Chopfee Coffee Shop - EOQ & ROP Analyzer | Developed for Inventory Optimization")