import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import json
from skforecast.model_selection_multiseries import grid_search_forecaster_multivariate
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from connection import view_queries


weeklySales2=view_queries.get_sales_weekly()
weeklySales2.info(show_counts=True)

weeklySales2[['SumaCostoSemanal','SumaUtilidadSemanal','SumaPorcentSemanal']]=weeklySales2[['SumaCostoSemanal','SumaUtilidadSemanal','SumaPorcentSemanal']].astype(float)

weeklySales2.head()

weeklySales2['puti']=weeklySales2['SumaUtilidadSemanal']/weeklySales2['SumaCostoSemanal']
weeklySales2['puti']=weeklySales2['puti'].fillna(0)
weeklySales2.head()


scaler3 = StandardScaler()

ventas_num_ = ['SumaVentaSemanal', 'SumaCostoSemanal', 'SumaUtilidadSemanal',
       'puti', 'PromedioDiasSinVenta']

ventas2=weeklySales2.copy()
ventas2[ventas_num_] = scaler3.fit_transform(ventas2[ventas_num_])   #Solo escalar entradas, label es 'CantidadVendida'



datos_series=[]
productos=ventas2['productCode'].unique()
predicciones=[]
predicciones_fut=[]          
models=[]      # mse train: models[i]['mean_squared_error']
mse_test=[]

error_codes=[]


# param_grid={'n_estimators':[50,80,100],'max_depth':[2,5,10],'min_samples_leaf':[5,10]}
param_grid={'n_neighbors':[2,3,4,5,6,7,8]}     #param grid for cv - gridsearch

for i,product in enumerate(ventas2['productCode'].unique()):
    # print(f"product{i}: {product}")
    # if i>2:
    #     break
    product_data = ventas2[ventas2['productCode'] == product]
    product_data=product_data.set_index('FechaLunesSemana')
    # product_data=product_data.asfreq('W')
    product_data=product_data.sort_index()

    datos_series.append(product_data)   # append to series list by product for visualization

    steps=12    # weekly ->3months is 12 steps
    datos_train=product_data[:-steps]
    datos_test=product_data[-steps:]
    # print(datos_train.head())
    # print(datos_train.shape)
    
    try:
        forecaster2=ForecasterAutoregMultiVariate(
                        regressor=KNeighborsRegressor(),
                        level='SumaCantidadSemanal',
                        steps=steps,
                        lags=1
                    )

        
        results_grid=grid_search_forecaster_multivariate(
                        forecaster=forecaster2,
                        series=datos_train[['PromedioDiasSinVenta','SumaCantidadSemanal','puti','SumaUtilidadSemanal']],
                        param_grid=param_grid,
                        steps=steps,
                        refit=True,
                        metric='mean_squared_error',
                        initial_train_size=len(datos_train[:-steps*2]),
                        fixed_train_size=False,
                        return_best=True,
                        verbose=False
                    )

        results_grid.reset_index(drop=True,inplace=True)

        models.append(results_grid.loc[0])

        #predictions test
        predictions_cv=forecaster2.predict_interval()
        predictions_cv['Codigo']=product
        predictions_cv.reset_index(names=['Dates'],inplace=True)
        predictions_cv['Dates']=datos_train.index.to_list()[-1]+pd.to_timedelta(predictions_cv['Dates'],unit='W')
        # print(predictions_cv)
        predicciones.append(predictions_cv)

        mse = mean_squared_error(datos_test['SumaCantidadSemanal'],predictions_cv['SumaCantidadSemanal'])
        mse_test.append(mse)

        #predictions future
        predictions_future=forecaster2.predict_interval(last_window=datos_test[['PromedioDiasSinVenta','SumaCantidadSemanal','puti','SumaUtilidadSemanal']])
        predictions_future['Codigo']=product
        predictions_future.reset_index(names=['Dates'],inplace=True)
        predictions_future['Dates']=datos_train.index.to_list()[-1]+pd.to_timedelta(predictions_future['Dates'],unit='W')
        # print(predictions_future)
        predicciones_fut.append(predictions_future)


    except:
        error_codes.append(product)


code_names=view_queries.get_product_sold_select_all()

data_test = {}
data_pred = {}

for i in range(len(predicciones)):
    codigo = predicciones[i]['Codigo'][0]    #predicciones [df[pred,pred5,pred95,fecha,Codigo],df,df,....]

    nombre = code_names.loc[code_names['productCode']==codigo,'name'].iloc[0]  #get name from code

    if nombre not in data_test:
        data_test[nombre] = {'mse': "",
                        'n_neighbors':"",
                        'time_series_dates': [],
                        'time_series_values':[],
                        'prediction_dates': [],
                        'prediction_values':[],
                        'confianza5': [],
                        'confianza95':[],
                        'codigo': "",
                        'nombre': ""}
    if nombre not in data_pred:
        data_pred[nombre] = {'time_series_dates': [],
                        'time_series_values':[],
                        'prediction_dates': [],
                        'prediction_values':[],
                        'confianza5': [],
                        'confianza95':[],
                        'codigo': "",
                        'nombre': ""}

    #test results    
    data_test[nombre]['mse']=str(models[i]['mean_squared_error'])
    data_test[nombre]['n_neighbors']=str(models[i]['n_neighbors'])
    
    data_test[nombre]['time_series_values'].extend(datos_series[i]['SumaCantidadSemanal'].values.tolist())
    fechas = datos_series[i].index.to_list()
    fechas_str = [fecha.strftime('%Y-%m-%d %H:%M:%S') for fecha in fechas]
    data_test[nombre]['time_series_dates'].extend(fechas_str)

    data_test[nombre]['prediction_dates'].extend([d.split('T')[0] for d in predicciones[i]['Dates'].values.astype(str)])
    data_test[nombre]['prediction_values'].extend(predicciones[i]['SumaCantidadSemanal'].values.tolist())
    data_test[nombre]['confianza5'].extend(predicciones[i]['lower_bound'].values.tolist())
    data_test[nombre]['confianza95'].extend(predicciones[i]['upper_bound'].values.tolist())

    data_test[nombre]['codigo']=codigo
    data_test[nombre]['nombre']=nombre
    
    #prediction results
    data_pred[nombre]['time_series_values'].extend(datos_series[i]['SumaCantidadSemanal'].values.tolist())
    fechas = datos_series[i].index.to_list()
    fechas_str = [fecha.strftime('%Y-%m-%d %H:%M:%S') for fecha in fechas]
    data_pred[nombre]['time_series_dates'].extend(fechas_str)

    data_pred[nombre]['prediction_dates'].extend([d.split('T')[0] for d in predicciones_fut[i]['Dates'].values.astype(str)])
    data_pred[nombre]['prediction_values'].extend(predicciones_fut[i]['SumaCantidadSemanal'].values.tolist())
    data_pred[nombre]['confianza5'].extend(predicciones_fut[i]['lower_bound'].values.tolist())
    data_pred[nombre]['confianza95'].extend(predicciones_fut[i]['upper_bound'].values.tolist())

    data_pred[nombre]['codigo']=codigo
    data_pred[nombre]['nombre']=nombre

# Guardar el diccionario en un archivo JSON
with open('jc_model_results_test.json', 'w') as f:
    json.dump(data_test, f)

with open('jc_model_results_future.json', 'w') as f:
    json.dump(data_pred, f)