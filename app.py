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

st.title('Google Trends Populariteit')

landen = {"Nederland": "NL"}
periodes = {
    "Laatste maand": "today 1-m",
    "Laatste 3 maanden": "today 3-m",
    "Laatste 12 maanden": "today 12-m",
    "Alles": "all"
}

with st.form(key='zoek_formulier'):
    zoekwoorden_input = st.text_input(
        'Voer 1 tot 5 zoekwoorden in, gescheiden door komma:',
        'appelboom, sinaasappel, peer'
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        land = st.selectbox('Kies een land/regio:', list(landen.keys()))
    with col2:
        gebruik_aangepaste_range = st.checkbox("Gebruik aangepaste datumrange")

    if gebruik_aangepaste_range:
        vandaag = date.today()
        col3, col4 = st.columns(2)
        with col3:
            startdatum = st.date_input("Startdatum", vandaag - timedelta(days=365))
        with col4:
            einddatum = st.date_input("Einddatum", vandaag)

        if startdatum >= einddatum:
            st.error("Startdatum moet vóór einddatum liggen.")
            geselecteerde_periode = None
        else:
            geselecteerde_periode = f"{startdatum} {einddatum}"
    else:
        periode = st.selectbox('Kies een vooraf ingestelde periode:', list(periodes.keys()))
        geselecteerde_periode = periodes[periode]

    zoekknop = st.form_submit_button("Zoek")

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

if zoekknop:
    if not geselecteerde_periode:
        st.warning("Selecteer een geldige periode.")
    else:
        zoekwoorden = [kw.strip() for kw in zoekwoorden_input.split(',') if kw.strip()]
        if len(zoekwoorden) == 0:
            st.warning("Voer minimaal één zoekwoord in.")
        elif len(zoekwoorden) > 5:
            st.warning("Je kunt maximaal 5 zoekwoorden tegelijk vergelijken.")
        else:
            try:
                with st.spinner('Data ophalen...'):
                    df = haal_trends_data_met_retry(zoekwoorden, geselecteerde_periode, landen[land])

                if df.empty:
                    st.warning("Geen data gevonden voor deze instellingen. Mogelijk ondersteunt Google deze combinatie van periode en land niet.")
                    st.info("Tip: Probeer een kortere periode of kies 'Worldwide'.")
                else:
                    df = df.rename(columns={'date': 'Datum'})
                    df_melted = df.melt(id_vars=['Datum'], value_vars=zoekwoorden,
                                        var_name='Zoekwoord', value_name='Populariteit')

                    # Hoofd grafiek
                    fig = px.line(df_melted, x='Datum', y='Populariteit', color='Zoekwoord',
                                  title=f'Populariteit van zoekwoorden in {land} over de geselecteerde periode',
                                  labels={'Datum': 'Datum', 'Populariteit': 'Populariteit', 'Zoekwoord': 'Zoekwoord'},
                                  range_y=[0, 100])

                    fig.update_layout(yaxis=dict(tick0=0, dtick=25),
                                      xaxis_tickangle=-45,
                                      template='plotly_white')

                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Bekijk diepgaande analyses per zoekwoord", expanded=False):
                        for woord in zoekwoorden:
                            st.markdown(f"### Zoekwoord: {woord}")

                            df_woord = df[['Datum', woord]].rename(columns={woord: 'Populariteit'}).dropna().copy()

                            col1, col2 = st.columns([2, 1])

                            with col1:
                                fig_woord = px.line(df_woord, x='Datum', y='Populariteit',
                                                    title=f'Trend voor: {woord}',
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
                                    title=f'Seizoensanalyse',
                                    xaxis_title='Maand',
                                    yaxis_title='Gemiddelde populariteit',
                                    yaxis=dict(range=[0, 100]),
                                    template='plotly_white',
                                    xaxis_tickangle=-45,
                                    showlegend=False,
                                    margin=dict(t=50, b=40)
                                )

                                st.plotly_chart(fig_seizoen, use_container_width=True)

                            piek_maand_naam = calendar.month_name[piek_maand_num]
                            start_campagne_maand_num = piek_maand_num - 1 if piek_maand_num > 1 else 12
                            start_campagne_naam = calendar.month_name[start_campagne_maand_num]

                            st.markdown(f"""
                            <div style="padding:15px; border-radius: 8px; background-color:#000000; color:#ffffff; margin-top:15px; margin-bottom:30px;">
                            <b>Let op:</b> '{woord}' piekt elk jaar in <b>{piek_maand_naam}</b>. Overweeg om je contentcampagne in <b>{start_campagne_naam}</b> te starten.
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Er is een fout opgetreden: {e}")
