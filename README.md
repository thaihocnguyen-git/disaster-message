# Disaster Response Pipeline Project

### Desciption
    This project create a machine learning pipeline using scikit-learn to classify message from
    disaster message dataset provided by [Figure Eight](https://www.figure-eight.com/).

    This classification can help to classify the message automatically and redirect the message
    to the organization that has responsibility to this disaster.

### Dataset
    The disaster dataset includes text messages about disasters that is labeled into 36 catergories:
        1. 'related'
        2. 'request'
        3. 'offer'
        4. 'aid_related'
        5. 'medical_help',
        6. 'medical_products'
        7. 'search_and_rescue'
        8. 'security'
        9. 'missing_people'
        10. 'refugees'
        11. 'death'
        12. 'other_aid'
        13. 'infrastructure_related'
        14. 'transport'
        15. 'buildings'
        16. 'electricity'
        17. 'tools'
        18. 'hospitals'
        19. 'shops'
        20. 'aid_centers'
        21. 'other_infrastructure'
        22. 'weather_related'
        23. 'floods'
        24. 'storm'
        25. 'fire'
        26. 'earthquake',
        27. 'cold'
        28. 'other_weather'
        29. 'direct_report'
        30. 'military'
        31. 'child_alone'
        32. 'water'
        33. 'food'
        34. 'shelter'
        35. 'clothing'
        36. 'money',
    (./images/categories_sum.PNG)
### Project structure
```
app                               **The web app folder**
|____templates                    The static resources for web app
|    |___go.html                  Page for predicting message
|    |___master.html              Home page
|____run.py                       Routers and entry point for web app
data                              **Contains dataset**
|____disaster_messages.csv        Raw data of messages
|____disaster_categories.csv      Raw data of categories
|____DisasterRepsponse.db         Processed data
|____process_data.py              Python script to process data
model                             **Contains model file and training script**
|____train_classifier.py          Python script to create, train and save model
|____classifier.pkl               Saved model
images                            **Images uses in readme**
requirements_conda.txt            Use to install anaconda environment
README.md
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
4. The website will be looked like
(./images/website1.PNG)
(./images/website2.PNG)

### Imbalance data
The dataset is imbalance. For example in the water label:
(./images/water.PNG)

### Further work
1. Research ways to handle [imbalance data](https://imbalanced-learn.org/stable/)
2. Design new feature, like Word2Vec, Continous Bag of Words, Glove
3. Use different models
### Acknowledgement
Thank you for your attetion in this project. Please don't hesitate to contact me <nguyenthaihoccantho@gmail.com> if you have any question or recommendation.