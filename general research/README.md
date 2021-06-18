	
(AI-PA)  AI ID Photo Authentication

A Folder holding all of the main Facial Recognition Reaserch done on the project 


1 Das, Avishek 

Racial bias is a major topic of concern when it comes to facial recognition. The training data for models can differ greatly depending on who and where the sample data was collected from. For example, a model trained in China may perform better with Asian facial features than a model developed in Africa. It may be impossible to completely eliminate bias, but we should try our best to understand it so that we can reduce it as much as possible.

	
If training our own models, try to get as much diversity in our training sets as we can. Sufficient data for different races, genders, facial features like high / low cheekbones, beards, etc.
Some bias is inevitable. When evaluating our models / or external libraries, go through a bias evaluation phase. Get samples of as many different races and face types as possible and understand how the our recognition performance is affected by different facial features so that we can tune parameters such as confidence intervals accordingly.



2 Carpenter, Dean A 

Bias is an important factor to consider when evaluating a facial recognition system, especially for our use cases with a lot of ethnically diverse faces. Every single facial recognition system has this to some extent, though which races they handle best vary. To combat this, a diverse selection of training data must be used, and it should be evaluated on different ethnic groups to identify any weak areas. This analysis needs to be done with the actual system and data that it will be encountering.

Fortunately for us, NIST did a large study of how a wide variety of algorithms performed (https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.8280.pdf), and the datasets they used included the ones we would be using as well - Immigration applications, visa applications, and border crossing photos. This should give us confidence that their results are meaningful to us. There is a stark difference between the best and worse options, with the best option proving to be VisionLabs, which had excellent scores across the board. *include VisionLabs pricing*.