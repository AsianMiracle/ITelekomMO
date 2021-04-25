from flask import render_template
from flask import request
from werkzeug.utils import secure_filename
import  matplotlib.pyplot as plt
import json
import os
from flask import Flask
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import datetime as dt


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'info'
    DEBUG = True
    MAX_CONTENT_PATH = 10000000
    UPLOAD_FOLDER = ""


app = Flask(__name__)
app.config.from_object(Config)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])

def exec():
    context = {}
    if request.method == 'POST':
        if request.form.get('rd_file') != None:
           #if request.form.get('rd_file') != None:

                df = pd.read_csv('upload_DJIA_table.csv')
                df2 = pd.read_csv('Combined_News_DJIA.csv')
                df2['Delta'] = df.Open - df.Close
                dff = pd.DataFrame({'Date':df2.Date, 'Label':df2.Label, 'Delta':df2.Delta})
                xs12 = dff[(dff.Date > "2008-08-08") & (dff.Date < "2016-07-01")].Date
                ys12 = dff[(dff.Date > "2008-08-08")  & (dff.Date < "2016-07-01")].Delta

                tryw = pd.DataFrame(np.array(ys12), index=range(len(xs12)))
                model = SARIMAX(tryw)
                model_fit = model.fit(disp=False)
                yhat = model_fit.predict(0, len(ys12))
                model_fit.aic,len(yhat)

                #df12 = pd.read_csv('Combined_News_DJIA.csv')
                train = df2[df2['Date']<'20150101']
                test = df2[df2['Date'] > '20141231']
                data = train.iloc[:, 2:27]
                data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
                list1 = [i for i in range(25)]
                new_Index = [str(i) for i in list1]
                data.columns = new_Index
                for index in new_Index:
                    data[index] = data[index].str.lower()
                headlines = []
                for row in range(0, len(data.index)):
                    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))
                countVector = CountVectorizer(ngram_range=(2,2))
                trainDataset = countVector.fit_transform(headlines)
                def pred_cl(df):
                    data = df.iloc[:, 2:27]
                    data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
                    list1 = [i for i in range(25)]
                    new_Index = [str(i) for i in list1]
                    data.columns = new_Index
                    for index in new_Index:
                        data[index] = data[index].str.lower()
                    headlines = []
                    for row in range(0, len(data.index)):
                        headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))
                    Dataset = countVector.transform(headlines)
                    return Dataset
                predictor_df = pred_cl(df2)
                with open('rf_b.pkl', 'rb') as f:
                    randomClassifier = pickle.load(f)
                    ch = randomClassifier.predict(predictor_df)
                lb = ch
                b = []
                for i in range(0, len(yhat)):
                    if lb[i] == 0 and yhat[i] < 0:
                        a = yhat[i]*(-1)
                        b.append(a)
                    elif lb[i] == 1 and yhat[i] > 0:
                        a = yhat[i]*(-1)
                        b.append(a)
                    else:
                        a = yhat[i]
                        b.append(a)
                dr = pd.DataFrame({'Date':df2.Date[:-1], 'Label':ch[:-1], 'Delta':b})
                modif_dr = dr[['Date', 'Delta']]

                context["val"] = modif_dr.Delta.tolist()
                context["dat"] = modif_dr.Date.tolist()
                context["f"] = modif_dr.columns.tolist()
                context["cw"] = modif_dr.values.tolist()[1:30]
                
                start_date = dt.datetime(2016, 6, 30)
                end_date = dt.datetime(2018, 7, 28)
                data_pred = pd.date_range(
                min(start_date, end_date),
                max(start_date, end_date)
                ).strftime('%Y-%m-%d').tolist()
                tryw1 = pd.DataFrame(np.array(dff.Delta), index=range(len(dff.Date)))
                model1 = SARIMAX(tryw1)
                model_fit1 = model1.fit(disp=False)
                yhat_pr = model_fit1.predict(0, len(data_pred))
                model_fit1.aic,len(yhat_pr)
                drr = pd.DataFrame({'Date':data_pred, 'Delta':yhat_pr[:-1]})
                context["dp_date"] = drr.values.tolist()[:30]
                context["dp_date_cols"] = drr.columns.tolist()
                context["dp_d"] = drr.Date.tolist()
                context["dp_v"] = drr.Delta.tolist()
                print(context["dp_v"])
                return render_template('index.html', ctx = context)
        
        if (request.form.get('date_from') != None) & (request.form.get('date_to') != None):
            fr = str(request.form.get('date_from'))
            fr = fr.split('.')
            fr1 = int(fr[0])
            fr2 = int(fr[1])
            fr3 = int(fr[2])
            start_date = dt.datetime(fr3, fr2, fr1)
            print(start_date)

            t = str(request.form.get('date_to'))
            t = t.split('.')
            t1 = int(t[0])
            t2 = int(t[1])
            t3 = int(t[2])
            end_date = dt.datetime(t3, t2, t1)
            print(end_date)
            
            data_pred = pd.date_range(
                min(start_date, end_date),
                max(start_date, end_date)
            ).strftime('%Y-%m-%d').tolist()
            df = pd.read_csv('upload_DJIA_table.csv')
            df2 = pd.read_csv('Combined_News_DJIA.csv')
            df2['Delta'] = df.Open - df.Close
            dff = pd.DataFrame({'Date':df2.Date, 'Label':df2.Label, 'Delta':df2.Delta})

            tryw1 = pd.DataFrame(np.array(dff.Delta), index=range(len(dff.Date)))
            model1 = SARIMAX(tryw1)
            model_fit1 = model1.fit(disp=False)
            yhat_pr = model_fit1.predict(0, len(data_pred))
            model_fit1.aic,len(yhat_pr)
            drr = pd.DataFrame({'Date':data_pred, 'Delta':yhat_pr[:-1]})
            context["dp_date"] = drr.values.tolist()[:30]
            context["dp_date_cols"] = drr.columns.tolist()
            context["dp_d"] = drr.Date.tolist()
            context["dp_v"] = drr.Delta.tolist()

            return render_template('index.html', ctx = context)

        return render_template('index.html', ctx = context)

                
    return render_template('index.html', ctx = context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=55003, debug=True)
