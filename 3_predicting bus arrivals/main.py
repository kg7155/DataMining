import linear
import datetime
import numpy as np
import pandas as pd

''' FUNCTIONS 
'''
# make strings uppercase and strip them
def makeStringsNicer (dataframe):
    for index, row in dataframe.iterrows():
        route_dir = row["RouteDirection"].upper()
        route_dir = route_dir.strip()
        dataframe.at[index, "RouteDirection"] = route_dir
    return dataframe

# merge the same routes into same RouteDirection names, ie. {[A - B], [B - A]}. [B - A] becomes [A - B].
def merge(dataframe):
    for index, row in dataframe.iterrows():
        route_dir = row["RouteDirection"].upper()
        if ("-" in route_dir):
            route_dir = route_dir.split("-")
            for i in range(0, len(route_dir)):
                route_dir[i] = route_dir[i].strip()
            route_dir = sorted(route_dir)
            dataframe.at[index, "RouteDirection"] = "".join(route_dir)
    return dataframe

# preprocess data
def preprocess(train_path, test_path):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    
    # read and convert train and test data to Pandas dataframe
    train = pd.read_csv(train_path, sep="\t", header=0, usecols=["Route", "RouteDirection","DepartureTime", "ArrivalTime"],
                            parse_dates=["DepartureTime", "ArrivalTime"], date_parser=dateparse)

    test = pd.read_csv(test_path, sep="\t", header=0, usecols=["Route", "RouteDirection", "DepartureTime"],
                           parse_dates=["DepartureTime"], date_parser=dateparse)

    # merge same routes into same RouteDirection names, ie. A - B, B - A. 
    # B - A becomes A - B.
    #train = merge(train)
    #test = merge(test)
    train = makeStringsNicer(train)
    test = makeStringsNicer(test)

    train["TravelTime"] = train["ArrivalTime"] - train["DepartureTime"]

    day_attr = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in day_attr:
        train[day] = train["DepartureTime"].dt.weekday_name == day
        test[day] = test["DepartureTime"].dt.weekday_name == day

    train["Weekend"] = (train["DepartureTime"].dt.weekday_name == "Saturday") | (train["DepartureTime"].dt.weekday_name == "Sunday")
    test["Weekend"] = (test["DepartureTime"].dt.weekday_name == "Saturday") | (test["DepartureTime"].dt.weekday_name == "Sunday")

    time_attr = list(range(1,24))
    for time in time_attr:
        train[time] = train["DepartureTime"].dt.hour == time
        test[time] = test["DepartureTime"].dt.hour == time

    work_free_days = [ [2012, 1, 1], [2012, 1, 2], [2012, 2, 8], [2012, 4, 8], [2012, 4, 9], [2012, 4, 27], [2012, 5, 1], 
                       [2012, 5, 2], [2012, 5, 31], [2012, 6, 25], [2012, 8, 15], [2012, 10, 31], [2012, 11, 1], [2012, 12, 25], [2012, 12, 26] ]

    for work_free_day in work_free_days:
        year = work_free_day[0]
        month = work_free_day[1]
        day = work_free_day[2]
        name = "" + str(year) + "-" + str(month) + "-" + str(day)
        train[name] = (train["DepartureTime"].dt.year == year) & (train["DepartureTime"].dt.month == month) & (train["DepartureTime"].dt.day == day)
        test[name] = (test["DepartureTime"].dt.year == year) & (test["DepartureTime"].dt.month == month) & (test["DepartureTime"].dt.day == day)
    
    '''
    month_attr = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    for month in range(1,13):
        train[month_attr[month-1]] = train["DepartureTime"].dt.month == month
        test[month_attr[month-1]] = test["DepartureTime"].dt.month == month
    '''

    train["RushHour"] = (train["DepartureTime"].dt.hour == 7) | (train["DepartureTime"].dt.hour == 8) | (train["DepartureTime"].dt.hour == 15) | (train["DepartureTime"].dt.hour == 16)
    test["RushHour"] = (test["DepartureTime"].dt.hour == 7) | (test["DepartureTime"].dt.hour == 8) | (test["DepartureTime"].dt.hour == 15) | (test["DepartureTime"].dt.hour == 16)
    
    return [train, test]

# train models with route directions
def train_route_dirs(route_dirs_to_train, train):
    models_dirs = {}
    routes_to_train = []
    for route_dir in route_dirs_to_train:
        X_train = train[train["RouteDirection"] == route_dir]

        # train later with route number
        if (X_train.shape[0] == 0):
            rows = train.loc[train["RouteDirection"] == route_dir, "Route"]
            if (len(rows) > 0):
                routes_to_train.append(rows.iloc[0])
            continue

        Y_train = X_train["TravelTime"].dt.total_seconds()
        X_train = X_train.drop(["TravelTime", "DepartureTime", "ArrivalTime", "Route", "RouteDirection"], axis=1)

        lr = linear.LinearLearner(lambda_=1.)
        models_dirs[route_dir] = lr(np.array(X_train), np.array(Y_train))

    return [routes_to_train, models_dirs]

# train models with route numbers
def train_routes(routes_to_train, train):
    models_routes = {}
    for route in routes_to_train:
        X_train = my_train[my_train["Route"] == route]
        Y_train = X_train["TravelTime"].dt.total_seconds()
        X_train = X_train.drop(["TravelTime", "DepartureTime", "ArrivalTime", "Route", "RouteDirection"], axis=1)
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        lr = linear.LinearLearner(lambda_=1.)
        models_routes[route] = lr(X_train, Y_train)
    return models_routes

# predict travel time for test data
def predict(row, index, models_dirs, models_routes, train):
    route_dir = row["RouteDirection"]
    route = row["Route"]
    row = row.drop(["DepartureTime", "Route", "RouteDirection"])
    row = np.array(row)

    if (route_dir in models_dirs):
        pred_time = models_dirs[route_dir](row)
    elif (route in models_routes):
        pred_time = models_routes[route](row)
    else:
        pred_time = train["TravelTime"].dt.total_seconds().mean()

    return pred_time

# evaluate predictions using MAE
def evaluate(train, month_index):
    # split train data to train and test
    my_test = train[train["DepartureTime"].dt.month == month_index]
    my_train = train[train["DepartureTime"].dt.month != month_index]
    
    # train models using train data above
    route_dirs_to_train = my_test.RouteDirection.unique()

    [routes_to_train, models_dirs] = train_route_dirs(route_dirs_to_train, my_train)
    models_routes = train_routes(routes_to_train, my_train)
    
    # predict and evaluate
    mae = 0
    for index, row in my_test.iterrows():
        dep_time = row["DepartureTime"]
        arrival_time = row["ArrivalTime"]
        row = row.drop(["ArrivalTime", "TravelTime"])
        pred_time = predict(row, index, models_dirs, models_routes, my_train)
        
        mae += abs((arrival_time - (dep_time + pd.to_timedelta(pred_time, unit='s'))).total_seconds())
        
    print("MAE: ", mae / my_test.shape[0])
    

''' MAIN
'''
if __name__ == "__main__":
    # path to train and test file
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    # preprocess
    [train, test] = preprocess(train_path, test_path)
    
    # find all route directions of test data
    route_dirs_to_train = test.RouteDirection.unique()

    # train models using train data above
    [routes_to_train, models_dirs] = train_route_dirs(route_dirs_to_train, train)
    models_routes = train_routes(routes_to_train, train)

    # my evaluate: function takes train data and the number of the month for test
    evaluate(train, 11)
        
    # predict travel time for all test rows
    with open("predictions_2.txt", "w") as f:
        for index, row in test.iterrows():
            dep_time = row["DepartureTime"]
            pred_time = predict(row, index, models_dirs, models_routes, train)
            arr_time = dep_time + pd.to_timedelta(pred_time, unit='s')
            print(arr_time.strftime("%Y-%m-%d %H:%M:%S.%f"), file=f)
    