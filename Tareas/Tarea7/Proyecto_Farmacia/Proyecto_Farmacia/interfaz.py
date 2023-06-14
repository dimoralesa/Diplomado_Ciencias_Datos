import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import json
@st.cache_data
def cargar_datos():
    df_predictions = pd.read_json('./model/jc_model_results_future.json', lines=False)
    df_predictions = df_predictions.transpose()

    # df_products = pd.read_csv("./model/prducts.csv", lineterminator=';', delimiter=',', encoding='utf-16')
    print('Cargando productos...')
    all_products = pd.read_csv('./csv/productDim.csv', lineterminator='\n', delimiter=',', encoding='utf-16')
    print("Productos Cargados")
    print("Cargando ventas...")
    all_sales = pd.read_csv('./csv/weekly_ready.csv')
    print("Ventas Cargadas")
    df_products = all_products

    return df_predictions ,df_products ,all_products ,all_sales 

df_predictions ,df_products ,all_products ,all_sales  =  cargar_datos()

result = pd.merge(df_predictions, df_products, left_on='codigo', right_on="codigo", how='left')

st.title('Droguería la Especial')
st.header('Prediccion de ventas')
st.subheader('Ingrese el nombre del producto que desea analizar')
select = st.selectbox('Puedes escribir el nómbre o seleccionar de la lista',options=(result['nombre']))
print(select)



st.subheader('Gráfica de ventas vs predicción')
with open('model/jc_model_results_future.json','r') as json_jc:
    data_json=json.load(json_jc)

# predictions = pd.read_csv('model/preicciones.csv')





#Modelo
series_data=pd.DataFrame(index=data_json[select]['time_series_dates'],data={'serie':data_json[select]['time_series_values']})

predict_data=pd.DataFrame(data={'pred':data_json[select]['prediction_values'],
                                'conf5':data_json[select]['confianza5'],
                                'conf95':data_json[select]['confianza95']})

#Removed index declaration in line above, compute dates from last found in the current data:
predict_data['Date']=pd.Timestamp(max(data_json[select]['time_series_dates']))+pd.to_timedelta(predict_data.index,unit='W')
predict_data['Date']=predict_data['Date'].astype(str)
predict_data.set_index('Date',inplace=True)

fig,ax=plt.subplots()
ax.plot(series_data['serie'],label='series')

ax.plot(predict_data['pred'],label='prediction')

ax.fill_between(x=predict_data.index,y1=predict_data['conf5'],y2=predict_data['conf95'],alpha=0.3)

ax.legend()
ax.set_xlabel("Date")
ax.set_xticks(data_json[select]['time_series_dates']+data_json[select]['prediction_dates'])
ax.set_ylabel("Predicted Sales")
plt.title(f"Forecast for {data_json[select]['nombre']}")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.show()

st.pyplot(fig)


#Metricas
ventas = all_sales
ventas['FechaReporte'] = pd.to_datetime(ventas['FechaReporte'])

ventas['UtilidadUnitaria'] = ventas.apply(lambda row: row['Utilidad'] / row['Cant Vendidas'] if row['Cant Vendidas'] != 0 else 0, axis=1)

ventas = ventas.sort_values('FechaReporte')
ultima_utilidad = ventas.groupby('codigo')['Utilidad'].last()
utilidades = pd.DataFrame(ultima_utilidad)
# df_products = pd.read_csv("./model/prducts.csv", lineterminator=';', delimiter=',', encoding='utf-16')
df_products = all_products
result2 = pd.merge(df_predictions, df_products, left_on='codigo', right_on="codigo", how='left')

cod = result2.loc[result2['nombre'] == select, 'codigo'].iloc[0]
result2 = pd.merge(result2, utilidades, left_on='codigo', right_on="codigo", how='inner')
if len(result2.loc[result2['nombre'] == select, 'Utilidad']) > 0:
    utilidadUnitaria = result2.loc[result2['nombre'] == select, 'Utilidad'].iloc[0]
else:
    utilidadUnitaria = 0  # O asigna un valor por defecto en caso de no encontrar el valor


st.markdown('---')#Applying markdown
st.subheader('Recomendación de inventario')
number = st.number_input('Ingrese stock de reserva', step=1)

variable = result.loc[result['nombre'] == select, 'TotalValueUnits\r'].iloc[0]
variable = variable.astype(int)
print(variable)
st.write(f'>Inventario actual: {variable}')

invPrediccion = data_json[select]['prediction_values']
suma = round(sum(invPrediccion))
print(suma)

rentabilidad12Semanas = round( suma * utilidadUnitaria)

recomendacion = suma + number - variable
st.subheader(f'Te recomendamos comprar {recomendacion} unidades del producto {select} ')
st.markdown('---')#Applying markdown
st.metric("Rentabilidad próximos 3 meses", f"{'{:,.2f}'.format(rentabilidad12Semanas)} COP")

