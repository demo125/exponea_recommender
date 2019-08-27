<h1>Exponea recommender</h1>
<p>
Goal of the assignment was to create personalized recommender system for serving personalized products for each customer in real-time and build a REST API service to fetch recommendations from model.
</p>
<p>
  I decided to build three simple recommender system models that work together to serve most relevant recommendations. The first model is based on user interaction with the system. The event dataset was used to create collaborative filtering based recommendation system which was trained, evaluated and compared to few baseline models. The well know cold-strat problem - problem with new customers and products with not enough interaction for collaborative filtering to work properly was solved with a model that uses product features. 
  </p>
  <p>In the real world problem there would be some customer features which could be used for making even more personalised recommendations for new customers. In this project I build several models that computes similarity of individual features which are then combined to single model that computes similarity between products. If a customer has only few interactions with the system, this model uses these interactions and finds n most similar items to items the customer had interacted with.
  </p>
  <p>
  Lastly the third and simplest model is build for customers with no interactions. Because of no customer features the model could do nothing but recommend the most popular products otherwise it would be possible to build model for finding similar users and use their preferences for the new user. In the real world we could use at least location of the customer which can be easily found from ip or we could use seasonalty of products.
  </p>
  <p>More detailed findings can be found in jupyter notebooks in /notebooks directory. There is one notebook for every model. Notebooks are also exported to HTML in the same directory which can be easily opened.
 </p>
  <p>Simple REST API is build for serving real-time recommendations. The api has only few endpoints for train the models, finding similar items and for making recommendations. Training of the first model takes around 20 seconds, the other two are almost instant.
  </p>
  <p>I used popular python libraries for data handling ang visualisation - numpy, pandas, matplotlib, sklearn and LightFM for collaborative filtering based model. The API is build in Flask and runs on its own development server. The application was build in anaconda enviroment. Package list is exported to requirements.txt. 
  </p>
