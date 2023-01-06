# slip12


Q.1 Write a Java Program to implement Decorator Pattern for interface Car to define the
assemble() method and then decorate it to Sports car and Luxury Car 
Q. 2 Write a python program to make Categorical values in numeric format for a given
dataset
Q.3 Create a Simple Web Server using node js.




Q.1 Write a Java Program to implement Decorator Pattern for interface Car to define the
assemble() method and then decorate it to Sports car and Luxury Car
:
interface Car {
public void assemble();
}
 class BasicCar implements Car {
 @Override
public void assemble() {
System.out.print("Basic Car.");
}
}
 class CarDecorator implements Car {
protected Car car;
public CarDecorator(Car c){
this.car=c;
}
@Override
public void assemble() {
this.car.assemble();
}
}
class SportsCar extends CarDecorator {
public SportsCar(Car c) {
super(c);
}
@Override
public void assemble(){
car.assemble();
System.out.print(" Adding features of Sports Car.");
}
} 
class LuxuryCar extends CarDecorator {
public LuxuryCar(Car c) {
super(c);
}
public void assemble(){
car.assemble();
System.out.print(" Adding features of Luxury Car.");
}
}
public class Main {
public static void main(String[] args) {
Car s1 = new SportsCar(new BasicCar());
s1.assemble();
Car s2 = new LuxuryCar(new BasicCar());
s2.assemble();
}
} 


Q. 2 Write a python program to make Categorical values in numeric format for a given
dataset
:
//Before we get started encoding the various values, we need to important the
//data and do some minor cleanups. Fortunately, pandas makes this straightforward:
import pandas as pd
import numpy as np

# Define the headers since the data does not have any
headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

# Read in the CSV file and convert "?" to NaN
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
                  header=None, names=headers, na_values="?" )
df.head()


//The final check we want to do is see what data types we have:
df.dtypes


//df.dtypes
//
symboling              int64
normalized_losses    float64
make                  object
fuel_type             object
aspiration            object
num_doors             object
body_style            object
drive_wheels          object
engine_location       object
wheel_base           float64
length               float64
width                float64
height               float64
curb_weight            int64
engine_type           object
num_cylinders         object
engine_size            int64
fuel_system           object
bore                 float64
stroke               float64
compression_ratio    float64
horsepower           float64
peak_rpm             float64
city_mpg               int64
highway_mpg            int64
price                float64
dtype: object


//Since this article will only focus on encoding the categorical variables,
//we are going to include only the 
//object
// columns in our dataframe. Pandas has a
//helpful 
//select_dtypes
// function which we can use to build a new dataframe
//containing only the object columns.
obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()


//Before going any further, there are a couple of null values in the data that
//we need to clean up.
obj_df[obj_df.isnull().any(axis=1)]


//For the sake of simplicity, just fill in the value with the number 4 (since that
//is the most common value):
obj_df["num_doors"].value_counts()


//obj_df["num_doors"].value_counts()
//
four    114
two      89
Name: num_doors, dtype: int64


//four    114
//two      89
//Name: num_doors, dtype: int64
//
obj_df = obj_df.fillna({"num_doors": "four"})


//We have already seen that the num_doors data only includes 2 or 4 doors. The
//number of cylinders only includes 7 values and they are easily translated to
//valid numbers:
obj_df["num_cylinders"].value_counts()


//obj_df["num_cylinders"].value_counts()
//
four      159
six        24
five       11
eight       5
two         4
twelve      1
three       1
Name: num_cylinders, dtype: int64


//Here is the complete dictionary for cleaning up the 
//num_doors
// and
//
//num_cylinders
// columns:
cleanup_nums = {"num_doors":     {"four": 4, "two": 2},
                "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}


//To convert the columns to numbers using 
//replace
//:
obj_df = obj_df.replace(cleanup_nums)
obj_df.head()


//The nice benefit to this approach is that pandas “knows” the types of values in
//the columns so the 
//object
// is now a 
//int64
//
obj_df.dtypes


//obj_df.dtypes
//
make               object
fuel_type          object
aspiration         object
num_doors           int64
body_style         object
drive_wheels       object
engine_location    object
engine_type        object
num_cylinders       int64
fuel_system        object
dtype: object


//One trick you can use in pandas is to convert a column to a category, then
//use those category values for your label encoding:
obj_df["body_style"] = obj_df["body_style"].astype('category')
obj_df.dtypes


//obj_df["body_style"] = obj_df["body_style"].astype('category')
//obj_df.dtypes
//
make                 object
fuel_type            object
aspiration           object
num_doors             int64
body_style         category
drive_wheels         object
engine_location      object
engine_type          object
num_cylinders         int64
fuel_system          object
dtype: object


//Then you can assign the encoded variable to a new column using the 
//cat.codes
// accessor:
obj_df["body_style_cat"] = obj_df["body_style"].cat.codes
obj_df.head()


//Hopefully a simple example will make this more clear. We can look at the column
//
//drive_wheels
// where we have values of 
//4wd
//, 
//fwd
// or 
//rwd
//.
//By using 
//get_dummies
// we can convert this to three columns with a 1 or 0 corresponding
//to the correct value:
pd.get_dummies(obj_df, columns=["drive_wheels"]).head()


//This function is powerful because you can pass as many category columns as you would like
//and choose how to label the columns using 
//prefix
//. Proper naming will make the
//rest of the analysis just a little bit easier.
pd.get_dummies(obj_df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"]).head()


//In this particular data set, there is a column called 
//engine_type
// that contains
//several different values:
obj_df["engine_type"].value_counts()


//obj_df["engine_type"].value_counts()
//
ohc      148
ohcf      15
ohcv      13
l         12
dohc      12
rotor      4
dohcv      1
Name: engine_type, dtype: int64


//For the sake of discussion, maybe all we care about is whether or not the engine
//is an Overhead Cam (OHC) or not. In other words, the various versions of OHC are all the same
//for this analysis. If this is the case, then we could use the 
//str
// accessor
//plus 
//np.where
// to create a new column the indicates whether or not the car
//has an OHC engine.
obj_df["OHC_Code"] = np.where(obj_df["engine_type"].str.contains("ohc"), 1, 0)


//The resulting dataframe looks like this (only showing a subset of columns):
obj_df[["make", "engine_type", "OHC_Code"]].head()


//For instance, if we want to do the equivalent to label encoding on the make of the car, we need
//to instantiate a 
//OrdinalEncoder
// object and 
//fit_transform
// the data:
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
obj_df["make_code"] = ord_enc.fit_transform(obj_df[["make"]])
obj_df[["make", "make_code"]].head(11)


//Scikit-learn also supports binary encoding by using the 
//OneHotEncoder.
//
//We use a similar process as above to transform the data but the process of creating
//a pandas DataFrame adds a couple of extra steps.
from sklearn.preprocessing import OneHotEncoder

oe_style = OneHotEncoder()
oe_results = oe_style.fit_transform(obj_df[["body_style"]])
pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_).head()


//The next step would be to join this data back to the original dataframe. Here is an example:
obj_df = obj_df.join(pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_))


//First we get a clean dataframe and setup the 
//BackwardDifferenceEncoder
//:
import category_encoders as ce

# Get a new clean dataframe
obj_df = df.select_dtypes(include=['object']).copy()

# Specify the columns to encode then fit and transform
encoder = ce.BackwardDifferenceEncoder(cols=["engine_type"])
encoder.fit_transform(obj_df, verbose=1).iloc[:,8:14].head()


//If we try a polynomial encoding, we get a different distribution of values used
//to encode the columns:
encoder = ce.PolynomialEncoder(cols=["engine_type"])
encoder.fit_transform(obj_df, verbose=1).iloc[:,8:14].head()


//Here is a very quick example of how to incorporate the 
//OneHotEncoder
// and 
//OrdinalEncoder
//
//into a pipeline and use 
//cross_val_score
// to analyze the results:
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# for the purposes of this analysis, only use a small subset of features

feature_cols = [
    'fuel_type', 'make', 'aspiration', 'highway_mpg', 'city_mpg',
    'curb_weight', 'drive_wheels'
]

# Remove the empty price rows
df_ml = df.dropna(subset=['price'])

X = df_ml[feature_cols]
y = df_ml['price']


//Now that we have our data, let’s build the column transformer:
column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                                        ['fuel_type', 'make', 'drive_wheels']),
                                      (OrdinalEncoder(), ['aspiration']),
                                      remainder='passthrough')


//For the model, we use a simple linear regression and then make the pipeline:
linreg = LinearRegression()
pipe = make_pipeline(column_trans, linreg)


//Run the cross validation 10 times using the negative mean absolute error as our scoring
//function. Finally, take the average of the 10 values to see the magnitude of the error:
cross_val_score(pipe, X, y, cv=10, scoring='neg_mean_absolute_error').mean().round(2)



q3)Create a Simple Web Server using node js. 
:
var http = require('http'); // 1 - Import Node.js core module
var server = http.createServer(function (req, res) { // 2 -
creating server
 //handle incomming requests here..
});
server.listen(5000); 



