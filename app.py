import streamlit as st
from pytrends.request import TrendReq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np
import statsmodels.api as sm
from requests.exceptions import HTTPError
from datetime import date, timedelta
import calendar

st.set_page_config(layout="centered", page_title="Google Trends Populariteit")
st.title('📈 Google Trends Populariteit')

landen = {"Nederland": "NL"}
periodes = {
    "Laatste maand": "today 1-m",
    "Laatste 3 maanden": "today 3-m",
    "Laatste 12 maanden": "today 12-m",
    "Alles": "all",
    "Aangepaste periode": "custom"
}

# === Zoekwoordenveld direct bovenaan ===
zoekwoorden_input = st.text_input(
    '🔍 Voer 1 tot 5 zoekwoorden in, gescheiden door komma:',
    value='',
    placeholder='voorbeeld: fiets, vakantie, zonnebrand'
)

# === Land en periode ===
col1, col2 = st.columns([2, 1])
with col1:
    land = st.selectbox('🌍 Kies een land/regio:', list(landen.keys()))
with col2:
    periode = st.selectbox('🗓️ Kies een periode:', list(periodes.keys()))

# === Toon kalender als aangepaste periode gekozen is ===
startdatum = None
einddatum = None
if periode == "Aangepaste periode":
    col3, col4 = st.columns(2)
    with col3:
        startdatum = st.date_input("Begindatum", value=date.today() - timedelta(days=90))
    with col4:
        einddatum = st.date_input("Einddatum", value=date.today())

# === Submitknop ===
zoekknop = st.button("📊 Zoek")

# === Functie: Trends ophalen ===
def haal_trends_data_met_retry(zoekwoorden, timeframe, geo, max_retries=5, delay=2):
    pytrends = TrendReq(hl='nl-NL', tz=360)
    poging = 0
    while poging < max_retries:
        try:
            time.sleep(delay)
            pytrends.build_payload(zoekwoorden, cat=0, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if 'isPartial' in df.columns:
                df = df[~df['isPartial']]
            return df.reset_index()
        except HTTPError as e:
            if e.response.status_code == 429:
                poging += 1
                wachttijd = delay * (2 ** poging)
                st.warning(f"Te veel verzoeken (429). Probeer opnieuw over {wachttijd} seconden... (poging {poging}/{max_retries})")
                time.sleep(wachttijd)
            else:
                raise e
    raise Exception("Te vaak te snel verzoeken gedaan. Probeer het later opnieuw.")

# === Zoekactie ===
if zoekknop:
    zoekwoorden = [kw.strip() for kw in zoekwoorden_input.split(',') if kw.strip()]
    
    if len(zoekwoorden) == 0:
        st.warning("⚠️ Voer minimaal één zoekwoord in.")
    elif len(zoekwoorden) > 5:
        st.warning("⚠️ Je kunt maximaal 5 zoekwoorden tegelijk vergelijken.")
    else:
        try:
            # Tijdframe bepalen
            if periode == "Aangepaste periode":
                if not startdatum or not einddatum:
                    st.error("❌ Vul zowel een begin- als einddatum in.")
                    st.stop()
                if startdatum >= einddatum:
                    st.error("❌ De begindatum moet vóór de einddatum liggen.")
                    st.stop()
                timeframe = f"{startdatum.strftime('%Y-%m-%d')} {einddatum.strftime('%Y-%m-%d')}"
            else:
                timeframe = periodes[periode]

            with st.spinner('📡 Data ophalen van Google Trends...'):
                df = haal_trends_data_met_retry(zoekwoorden, timeframe, landen[land])

            if df.empty:
                st.warning("⚠️ Geen data gevonden voor deze instellingen.")
            else:
                df = df.rename(columns={'date': 'Datum'})
                df_melted = df.melt(id_vars=['Datum'], value_vars=zoekwoorden,
                                    var_name='Zoekwoord', value_name='Populariteit')

                fig = px.line(df_melted, x='Datum', y='Populariteit', color='Zoekwoord',
                              title=f'Populariteit van zoekwoorden in {land}',
                              labels={'Datum': 'Datum', 'Populariteit': 'Populariteit'},
                              range_y=[0, 100])

                fig.update_layout(yaxis=dict(tick0=0, dtick=25),
                                  xaxis_tickangle=-45,
                                  template='plotly_white')

                st.plotly_chart(fig, use_container_width=True)

                # === Verdieping per zoekwoord ===
                with st.expander("📌 Verdieping per zoekwoord"):
                    for woord in zoekwoorden:
                        st.markdown(f"### 🔎 Zoekwoord: `{woord}`")

                        df_woord = df[['Datum', woord]].rename(columns={woord: 'Populariteit'}).dropna().copy()

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            # Trendlijn
                            fig_woord = px.line(df_woord, x='Datum', y='Populariteit',
                                                title=f"Trend voor: {woord}",
                                                labels={'Datum': 'Datum', 'Populariteit': 'Populariteit'})

                            df_woord['Tijd'] = np.arange(len(df_woord))
                            X = sm.add_constant(df_woord['Tijd'])
                            model = sm.OLS(df_woord['Populariteit'], X).fit()
                            df_woord['Trendlijn'] = model.predict(X)

                            fig_woord.add_trace(go.Scatter(
                                x=df_woord['Datum'],
                                y=df_woord['Trendlijn'],
                                mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='Trendlijn',
                                hoverinfo='skip',
                                showlegend=False
                            ))

                            fig_woord.update_layout(
                                yaxis=dict(tick0=0, dtick=25),
                                xaxis_tickangle=-45,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_woord, use_container_width=True)

                        with col2:
                            # Seizoensanalyse
                            df_woord['Maand'] = df_woord['Datum'].dt.month
                            seizoen = df_woord.groupby('Maand')['Populariteit'].mean().reset_index()
                            seizoen['MaandNaam'] = seizoen['Maand'].apply(lambda x: calendar.month_name[x])
                            seizoen = seizoen.sort_values('Maand')

                            standaard_blauw = '#1f77b4'
                            licht_blauw = '#aec7e8'
                            piek_maand_num = seizoen.loc[seizoen['Populariteit'].idxmax(), 'Maand']
                            kleuren = [licht_blauw if maand == piek_maand_num else standaard_blauw for maand in seizoen['Maand']]

                            fig_seizoen = go.Figure(data=[go.Bar(
                                x=seizoen['MaandNaam'],
                                y=seizoen['Populariteit'],
                                marker_color=kleuren
                            )])

                            fig_seizoen.update_layout(
                                title='Seizoensanalyse',
                                xaxis_title='Maand',
                                yaxis_title='Gem. populariteit',
                                yaxis=dict(range=[0, 100]),
                                template='plotly_white',
                                xaxis_tickangle=-45,
                                showlegend=False
                            )

                            st.plotly_chart(fig_seizoen, use_container_width=True)

                        # Campagne-advies
                        piek_maand_naam = calendar.month_name[piek_maand_num]
                        start_campagne_maand_num = piek_maand_num - 1 if piek_maand_num > 1 else 12
                        start_campagne_naam = calendar.month_name[start_campagne_maand_num]

                        st.markdown(f"""
                        <div style="padding:15px; border-radius: 8px; background-color:#000000; color:#ffffff; margin-top:10px;">
                        📌 <b>Campagnetip:</b> '{woord}' piekt jaarlijks in <b>{piek_maand_naam}</b>. Overweeg je contentcampagne te starten in <b>{start_campagne_naam}</b>.
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Er is een fout opgetreden: {e}")
