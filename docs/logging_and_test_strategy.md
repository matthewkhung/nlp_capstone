# Logging and Test Strategies

As this capstone is not a large scale production, usage will be limited. However, if scaled up, an effective logging
and testing strategy will be necessary for monitoring, and continuous integration and deployment.

## Logging

Having good logging capabilities can help with several aspects of the project. These include debugging functional
issues, spot checking predictions served to users, and monitoring performance including model or data drift.

As there is always risk of an application crashing, logging traces and what the application was doing will help in
diagnosing the failure and solving it. The application currently supports a debug flag that will either expose Python
exceptions via the web application (to facilitate debugging) or will hide it from the end user (for production use).
Additional efforts can be taken to log these failures into a database or storage for further debugging and traceability.
Additionally, the models and pipeline modules also actions taken such as training and evaluating which are mostly used
by the notebook.

Second, logging user behaviour is a good method for spot checking the application. The data can be manually reviewed 
by an analyst to determine if the model is still working. If a prediction has a confidence level attached to it, 
this score can be used to dictate which predictions to log. To further improve the model's prediction accuracy, a 
feedback mechanism where users can specify a wrong prediction (either false positive or false negative) can be 
implemented.

Third, by logging metrics including user generated error feedback, the model's performance can be monitored. 
Additional metrics to monitor that may help with ML Ops include request load, model response time, resource loads, 
etc.

## Testing

Unit testing helps with assessing traditional software functionality. As this application is fairly straightforward, 
unit testing was not implemented.

Model performance testing will assess whether a model is performing better than the previous model. This helps 
determine if a new model should supersede the previous. In the case of this project, this step was done manually.

Shadow deployment testing helps if application availability is a priority. When deploying a model, differences 
between the development environment and the production one can cause deployment issues. By Dockerizing the 
application, many issues are eliminated. However, some issues can still crop up. For example, when deploying this 
application to the NAS server, the Docker container crashed with a 132 error (illegal instruction) that is 
usually caused by running the container on an older CPU. In this case, TensorFlow uses AVX instructions which are not 
supported on Celeron processors. A shadow deployment would catch this error before the production application is 
taken down.