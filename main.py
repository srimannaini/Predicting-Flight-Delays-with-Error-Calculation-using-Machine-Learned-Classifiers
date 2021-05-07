from flask import Flask, render_template, request, url_for, flash, Response
import os
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import shutil
#from flask.ext.cache import Cache
#from flask_caching import Cache

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, median_absolute_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#cache = Cache(config={"CACHE_TYPE": "simple"})
app=Flask(__name__)
#app.config['CACHE_TYPE']='null'
#cache.init_app(app)
app.config['UPLOAD_FOLDER']=r"D:\takeoffs\proj2\TK11012\Predicting Flight Delays with Error Calculation using Machine Learned Classifiers\CODE\uploads"
app.secret_key='4d6fffa3fa1f8937a12b40c6c244b9dd'

# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
#app.cache=Cache(app)

#with app.app_context():
#    cache.clear()

#Global Variables
full_data=None; df_processed=None; df_encoded=None
X=None; y1=None; y2=None
X_scaled=None
X_train=None; X_test=None; y_train=None; y_test=None
metric1={}; metric2={}
split1=False; split2=False
metric_names=['Mean Squared Error', 'Mean Absolute Error', 'Explained Variance Score', 'Median Absolute Error', 'R2 Score']

#Global functions
def load_data(path):
    fn=pd.read_csv(path)
    return fn

def preprocessing_data(file):
    #junk features
    file.drop(['CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','WEATHER_DELAY'], axis=1, inplace=True)
    #feature selection
    cols_to_keep=['DAY','AIRLINE','FLIGHT_NUMBER','DESTINATION_AIRPORT','ORIGIN_AIRPORT','DAY_OF_WEEK','TAXI_OUT','DEPARTURE_DELAY','ARRIVAL_DELAY']
    file=file[cols_to_keep]
    #null value removal
    file.dropna(axis=0, how="any", inplace=True)
    #converting mixed (object) datatypes to string
    for x in file.columns:
        if str(file[x].dtype)=='object':
            file[x]=file[x].astype('str')
    file.reset_index(drop=True, inplace=True)
    return file

def encoding_data(file):
    oe = OrdinalEncoder()
    df_obj=file.select_dtypes(include=['object'])
    fit=oe.fit_transform(df_obj)
    df_obj=pd.DataFrame(fit, columns=df_obj.columns)
    for x in df_obj.columns:
        file[x]=df_obj[x]
    return file

def splitting_dataset(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)
    return X_train, X_test, y_train, y_test

def all_metrics(y_true, y_pred):
    a=mean_squared_error(y_true, y_pred)
    b=mean_absolute_error(y_true, y_pred)
    c=explained_variance_score(y_true, y_pred)
    d=median_absolute_error(y_true, y_pred)
    e=r2_score(y_true, y_pred)
    return [a,b,c,d,e]

def visualizations(metric_id=0):
    t1 = pd.DataFrame(metric1).transpose()
    t1.columns = metric_names
    t2 = pd.DataFrame(metric2).transpose()
    t2.columns = metric_names

    s1 = pd.DataFrame(t1.iloc[:, metric_id])
    s2 = pd.DataFrame(t2.iloc[:, metric_id])

    s = pd.concat([s1, s2], axis=0)
    s = s.stack().reset_index()
    s.columns = ['Model', 'Y', metric_names[metric_id]]
    s['Target'] = ['Departure Delay'] * 4 + ['Arrival Delay'] * 4
    return s

# @app.after_request
# def add_header(response):
#     # response.cache_control.no_store = True
#     response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '-1'
#     return response

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/load_dataset', methods=["POST","GET"])
def load_dataset():
    if request.method=="POST":
        myfile=request.files['filename']
        ext=os.path.splitext(myfile.filename)[1]
        if ext.lower() == ".csv":
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.mkdir(app.config['UPLOAD_FOLDER'])
            myfile.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(myfile.filename)))
            flash('The data is loaded successfully','success')
            return render_template('load_dataset.html')
        else:
            flash('Please upload a CSV type document only','warning')
            return render_template('load_dataset.html')
    return render_template('load_dataset.html')

@app.route('/view_data')
def view_data():
    #dataset
    myfile=os.listdir(app.config['UPLOAD_FOLDER'])
    global full_data
    full_data=load_data(os.path.join(app.config["UPLOAD_FOLDER"], myfile[0]))
    temp=full_data.iloc[:50,:]
    full_data=full_data.sample(100000, replace=False)
    full_data.reset_index(drop=True, inplace=True)
    print(full_data.head())
    return render_template('view_data.html', col=temp.columns.values, df=list(temp.values.tolist()))

@app.route('/preprocessing')
def preprocessing():
    global df_processed
    df_processed=preprocessing_data(full_data)
    temp=df_processed.iloc[:50,:]
    return render_template('preprocess.html', col=temp.columns.values, df=list(temp.values.tolist()))

@app.route('/encoding')
def encoding():
    global df_encoded
    #Encoding Data
    df_encoded=encoding_data(df_processed)
    #Seperating Data
    global X, y1, y2
    X = df_encoded.iloc[:, :-2]
    y1 = df_encoded["DEPARTURE_DELAY"]
    y2 = df_encoded["ARRIVAL_DELAY"]
    #Scaling Data
    global X_scaled
    scale=StandardScaler()
    X_scaled=pd.DataFrame(scale.fit_transform(X), columns=X.columns)
    #Combined data
    df=pd.concat([X_scaled, y1, y2], axis=1)
    temp = df.iloc[:50, :]
    return render_template('encoding.html', col=temp.columns.values, df=list(temp.values.tolist()))

@app.route('/training', methods=["POST","GET"])
def training():
    if request.method=="POST":
        global metric1, metric2
        target_no=int(request.form['target'])
        model_no=int(request.form['algo'])
        global split1, split2
        global X_train, X_test, y_train, y_test
        if target_no==0:
            flash("You have not selected any target variable yet.","warning")
        elif target_no==1:
            #split data
            if split1==False:
                print(split1)
                print(split2)
                X_train, X_test, y_train, y_test = splitting_dataset(X_scaled, y1)
                split1=True
                split2=False
                print(split1)
                print(split2)
            #model data
            #print and store metrics
            if model_no==0:
                flash("You have not selected any model yet",'warning')
            elif model_no==1:
                dct = DecisionTreeRegressor(max_depth=12)
                dct.fit(X_train, y_train)
                y_pred = dct.predict(X_test)
                scores=all_metrics(y_test, y_pred)
                if metric1.get('Decision Tree')==None:
                    metric1['Decision Tree']=scores
                return render_template('training.html', scores=scores, name='Decision Tree', met=metric_names)
            elif model_no==2:
                br = BayesianRidge(normalize=False, verbose=5)
                br.fit(X_train, y_train)
                y_pred = br.predict(X_test)
                scores=all_metrics(y_test, y_pred)
                if metric1.get('Bayesian Ridge')==None:
                    metric1['Bayesian Ridge']=scores
                return render_template('training.html', scores=scores, name='Bayesian ridge', met=metric_names)
            elif model_no==3:
                rf = RandomForestRegressor(n_estimators=150, min_samples_split=3, max_features='sqrt', max_depth=17, min_samples_leaf=7, bootstrap=False, verbose=5)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                scores=all_metrics(y_test, y_pred)
                if metric1.get('Random Forest')==None:
                    metric1['Random Forest']=scores
                return render_template('training.html', scores=scores, name='Random Forest', met=metric_names)
            elif model_no==4:
                gb = GradientBoostingRegressor(n_estimators=25, verbose=5)
                gb.fit(X_train, y_train)
                y_pred = gb.predict(X_test)
                scores=all_metrics(y_test, y_pred)
                if metric1.get('Gradient Boosting')==None:
                    metric1['Gradient Boosting']=scores
                return render_template('training.html', scores=scores, name='Gradient Boosting', met=metric_names)
        elif target_no==2:
            #split data
            if split2==False:
                X_train, X_test, y_train, y_test = splitting_dataset(X_scaled, y2)
                split2=True
                split1=False
            #model data
            #print and store metrics
            if model_no==0:
                print("You have not selected any model yet")
            elif model_no==1:
                dct=DecisionTreeRegressor(max_depth=12)
                dct.fit(X_train, y_train)
                y_pred = dct.predict(X_test)
                scores=all_metrics(y_test, y_pred)
                if metric2.get('Decision Tree')==None:
                    metric2['Decision Tree']=scores
                return render_template('training.html', scores=scores, name='Decision Tree', met=metric_names)
            elif model_no==2:
                br = BayesianRidge(normalize=False, verbose=5)
                br.fit(X_train, y_train)
                y_pred = br.predict(X_test)
                scores=all_metrics(y_test, y_pred)
                if metric2.get('Bayesian Ridge')==None:
                    metric2['Bayesian Ridge']=scores
                return render_template('training.html', scores=scores, name='Bayesian Ridge', met=metric_names)
            elif model_no==3:
                rf = RandomForestRegressor(n_estimators=150, min_samples_split=3, max_features='sqrt', max_depth=17, min_samples_leaf=7, bootstrap=False, verbose=5)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                scores=all_metrics(y_test, y_pred)
                if metric2.get('Random Forest')==None:
                    metric2['Random Forest']=scores
                return render_template('training.html', scores=scores, name='Random Forest', met=metric_names)
            elif model_no==4:
                gb = GradientBoostingRegressor(n_estimators=25, verbose=5)
                gb.fit(X_train, y_train)
                y_pred = gb.predict(X_test)
                scores=all_metrics(y_test, y_pred)
                if metric2.get('Gradient Boosting')==None:
                    metric2['Gradient Boosting']=scores
                print("hooooooooooooooooooo")
                print(pd.DataFrame(metric1).transpose())
                print("pooooooooooooooooooooo")
                print(pd.DataFrame(metric2).transpose())
                return render_template('training.html', scores=scores, name='Gradient Boosting', met=metric_names)
    return render_template('training.html')

@app.route('/tabulation', methods=["POST","GET"])
def tabulation():
    if request.method=="POST":
        tab_no=int(request.form['mytab'])
        if tab_no==1:
            t=pd.DataFrame(metric1).transpose()
            t.columns=metric_names
        elif tab_no==2:
            t=pd.DataFrame(metric2).transpose()
            t.columns=metric_names
        return render_template('tabulation.html', col=metric_names, rows=list(t.values.tolist()), models=list(t.index.values))
    return render_template('tabulation.html')

@app.route('/makeviz', methods=["POST","GET"])
def makeviz():
    if request.method=="POST":
        met_no=int(request.form['mytab'])
        # with app.app_context():
        #     cache.clear()
        s=visualizations(met_no-1)
        #sns.set(rc={'figure.figsize': (20, 10)})
        plt.figure(figsize=(14,11))
        ax=sns.barplot(x="Model", y=metric_names[met_no-1], hue="Target", data=s)
        ax.legend_=None
        ax.tick_params(labelsize=16)
        ax.set_xlabel(ax.get_xlabel(),fontsize=18)
        ax.set_ylabel(ax.get_ylabel(),fontsize=18)
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
        # ax.set_ytick
        fig = ax.get_figure()
        fig.legend(fontsize=20)
        fig.savefig('static/images/plot.png')
        fig.clf()
        del ax
        return render_template('visual.html', url="static/images/plot.png")
    return render_template('visual.html')





# sns.barplot(t1.index)
# a=pd.DataFrame(t1.iloc[:,0])
# a['Model']=a.index
# sns.barplot(data=a, x="Model", y="Mean Squared Error")
# ax11=sns.barplot(x=t1.index.to_list(), y=list(t1['Mean Squared Error']))
# ax11.set(xlabel="Model", ylabel="Mean Squared Error (MSE)")
# sns.barplot(x='X', y='Z', hue='target', data=u)



if __name__=='__main__':
    app.run(debug=False)