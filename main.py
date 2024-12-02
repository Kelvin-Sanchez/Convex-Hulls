import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def data_generation(size, seed, fing_md, wrist_md):

    # Set numpy rand seed
    np.random.seed(1234)
    
    # Create a normal guassian distribution of data points for finger length and wrist size
    finger_length = np.random.normal(fing_md, 1, size)
    wrist_size = np.random.normal(wrist_md, 0.5, size)

    # Introduce correlation
    cov_matrix = np.array([[1, 0.7], [0.7, 1]])
    correlated_data = np.random.multivariate_normal([0, 0], cov_matrix, size)

    # Combine with original data
    combined_data = np.hstack((finger_length[:, np.newaxis], wrist_size[:, np.newaxis], correlated_data))
    combined_data = combined_data + np.array([10, 7, 0, 0])

    # Generate height based on a linear model (adjust coefficients as needed)
    height = 0.8 * combined_data[:, 0] + 0.6 * combined_data[:, 1] + 60

    # Create a DataFrame
    df = pd.DataFrame({'finger_length': combined_data[:, 0], 'wrist_size': combined_data[:, 1], 'height': height})
    height_label = []

    # Separate heights into bins as labels
    for height in df['height']:
        if height < 83:
            height_label.append('very small')
        elif height > 83 and height < 84:
            height_label.append('small')
        elif height > 84 and height < 85:
            height_label.append('average')
        elif height > 85 and height < 86:
            height_label.append('large')
        elif height > 86:
            height_label.append('very large')
    # Drop the old height values as they are no longer needed
    df.drop('height', axis=1, inplace=True)
    # Add the new labels created
    df['height_labels'] = height_label
    return df


def cv_plot(df1, df2):
    points = np.array([df1['finger_length'], df1['wrist_size']]).T
    hull = ConvexHull(points)
    convex_hull_points = points[hull.vertices]

    points2 = np.array([df2['finger_length'], df2['wrist_size']]).T
    hull2 = ConvexHull(points2)
    convex_hull_points2 = points2[hull2.vertices]

    polygon1 = Polygon(convex_hull_points)
    polygon2 = Polygon(convex_hull_points2)

    plt.scatter(df1['finger_length'], df1['wrist_size'], label='Data Points')
    plt.scatter(df2['finger_length'], df2['wrist_size'], label='Data Points')
    plt.plot(polygon1.exterior.xy[0], polygon1.exterior.xy[1], label='Data Set 1', color='red')
    plt.plot(polygon2.exterior.xy[0], polygon2.exterior.xy[1], label='Data Set 2', color='green')
    plt.title("Convex Hulls")
    plt.xlabel('Finger Length')
    plt.ylabel('Wrist Size')
    plt.legend()
    plt.autoscale(enable=True)
    plt.show()


def train_model(df):
    X = df[['finger_length', 'wrist_size']]
    y = df['height_labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a classification MLP model for training
    model = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=2000)

    # Train the model on the provided pandas dataframe
    model.fit(X_train, y_train)
    return model


def model_metrics(model, X_test, y_test):
    # make predictions on provided input
    y_pred = model.predict(X_test)

    # Print out metrics and return the accuracy score for dataset shift detection
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("F1-score: ", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return accuracy_score(y_test, y_pred)


def convex_hull_subtractions(df1, df2):

    # obtain the points from dataframe values
    points1 = df1[['finger_length', 'wrist_size']].values
    points2 = df2[['finger_length', 'wrist_size']].values

    # create convex hulls from points
    hull1 = ConvexHull(points1)
    hull2 = ConvexHull(points2)

    # create polygon from the vertices in hulls
    polygon1 = Polygon(points1[hull1.vertices])
    polygon2 = Polygon(points2[hull2.vertices])

    # calculate the overlap between the two polygons, original dataset - second dataset
    remove_overlap = polygon1.difference(polygon2)
    # Calculate the difference in area
    diff_area = remove_overlap.area

    num_vertices1 = len(hull1.vertices)
    num_vertices2 = len(hull2.vertices)
    print("Dataset1 # of vertices1: ", num_vertices1)
    print("Dataset2 # of vertices2: ", num_vertices2)
    return diff_area


def accuracy_change(model, df1, df2):
    X = df1[['finger_length', 'wrist_size']]
    y = df1['height_labels']

    X2 = df2[['finger_length', 'wrist_size']]
    y2 = df2['height_labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

    acc1 = model_metrics(model, X_train, y_train)
    acc2 = model_metrics(model, X_train2, y_train2)
    print("Initial Accuracy: ", acc1*100)
    print("Second Accuracy: ", acc2*100)
    return (acc2 - acc1) * 100


def run_test(sizes):
    for size in sizes:
        df1 = data_generation(size, 42, 10, 7)
        df2 = data_generation(size, 43, 12, 5)
        cv_plot(df1, df2)
        model1 = train_model(df1)
        print("Testing Sample Size: ", size, "************************************")
        print("Change in Accuracy: ", accuracy_change(model1, df1, df2))
        print("Area difference between Convex Hulls: ", convex_hull_subtractions(df1, df2))
        print("***************************************************************")


sample_sizes = [20000, 25000, 50000, 100000]

run_test(sample_sizes)
