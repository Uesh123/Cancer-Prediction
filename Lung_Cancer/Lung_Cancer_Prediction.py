import gradio as gr
import pandas as pd
import numpy as np
from joblib import load

def predict_Level(
        Age, Gender, Air_Pollution, Alcohol_use,
        Dust_Allergy, OccuPational_Hazards, Genetic_Risk,
        chronic_Lung_Disease, Balanced_Diet, Obesity, Smoking,
        Passive_Smoker, Chest_Pain, Coughing_of_Blood, Fatigue,
        Weight_Loss, Shortness_of_Breath, Wheezing,Swallowing_Difficulty, 
        Clubbing_of_Finger_Nails, Frequent_Cold,Dry_Cough, Snoring                
):
    
# load model
    model = load('Lung_cancer_prediction.joblib')

    # Create a dict array from the parameters
    data = {
        'Age': [Age],
        'Gender': [Gender],
        'Air Pollution': [Air_Pollution],
        'Alcohol use': [Alcohol_use],
        'Dust Allergy': [Dust_Allergy],
        'OccuPational Hazards': [OccuPational_Hazards],
        'Genetic Risk': [Genetic_Risk],
        'chronic Lung Disease': [chronic_Lung_Disease],
        'Balanced Diet': [Balanced_Diet],
        'Obesity': [Obesity],
        'Smoking': [Smoking],
        'Passive Smoker': [Passive_Smoker],
        'Chest Pain': [Chest_Pain],
        'Coughing of Blood': [Coughing_of_Blood],
        'Fatigue': [Fatigue],
        'Weight Loss': [Weight_Loss],
        'Shortness of Breath': [Shortness_of_Breath],
        'Wheezing': [Wheezing],
        'Swallowing Difficulty': [Swallowing_Difficulty],
        'Clubbing of Finger Nails': [Clubbing_of_Finger_Nails],
        'Frequent Cold': [Frequent_Cold],
        'Dry Cough': [Dry_Cough],
        'Snoring': [Snoring],
    }
    Xinp = pd.DataFrame(data)
    print(Xinp)

    # Predict the level
    Level = model.predict(Xinp)

    # return the level
    return Level[0]

# Create the gradio interface

ui = gr.Interface(
    fn = predict_Level,
    inputs = [
        gr.inputs.Textbox(placeholder='Age', default="33", numeric=True,label='Age'),
        gr.inputs.Textbox(placeholder='Gender', default="1", numeric=True,label='Gender'), 
        gr.inputs.Textbox(placeholder='Air_Pollution', default="2",numeric=True,label='Air Pollution'), 
        gr.inputs.Textbox(placeholder='Alcohol_use', default="4",numeric=True,label='Alcohol use'),
        gr.inputs.Textbox(placeholder='Dust_Allergy', default="5",numeric=True,label='Dust Allergy'),
        gr.inputs.Textbox(placeholder='OccuPational_Hazards', default="4",numeric=True,label='OccuPational Hazards'),
        gr.inputs.Textbox(placeholder='Genetic_Risk', default="3",numeric=True,label='Genetic Risk'),
        gr.inputs.Textbox(placeholder='chronic_Lung_Disease', default="2",numeric=True,label='chronic Lung Disease'),
        gr.inputs.Textbox(placeholder='Balanced_Diet', default="2",numeric=True,label='Balanced Diet'),
        gr.inputs.Textbox(placeholder='Obesity', default="4", numeric=True,label='Obesity'),
        gr.inputs.Textbox(placeholder='Smoking', default="3", numeric=True,label='Smoking'),
        gr.inputs.Textbox(placeholder='Passive_Smoker', default="2", numeric=True,label='Passive Smoker'),
        gr.inputs.Textbox(placeholder='Chest_Pain', default="2", numeric=True,label='Chest Pain'),
        gr.inputs.Textbox(placeholder='Coughing_of_Blood', default="4", numeric=True,label='Coughing of Blood'),
        gr.inputs.Textbox(placeholder='Fatigue', default="3",numeric=True,label='Fatigue'),
        gr.inputs.Textbox(placeholder='Weight_Loss', default="4", numeric=True,label='Weight Loss'),
        gr.inputs.Textbox(placeholder='Shortness_of_Breath', default="2", numeric=True,label='Shortness of Breath'),
        gr.inputs.Textbox(placeholder='Wheezing', default="2",numeric=True,label='Wheezing'),
        gr.inputs.Textbox(placeholder='Swallowing_Difficulty', default="3", numeric=True,label='Swallowing Difficulty'),
        gr.inputs.Textbox(placeholder='Clubbing_of_Finger_Nails', default="1", numeric=True,label='Clubbing of Finger Nails'),
        gr.inputs.Textbox(placeholder='Frequent_Cold', default="2", numeric=True,label='Frequent Cold'),
        gr.inputs.Textbox(placeholder='Dry_Cough', default="3", numeric=True,label='Dry Cough'),
        gr.inputs.Textbox(placeholder='Snoring', default="4", numeric=True,label='Snoring'),
        
    ],
    outputs = "text",
)

if __name__ == "__main__":      
    ui.launch(share=False)