# MATH373_Spring2021_Project2_ammuth

1) project2multiclass_iris.py is the code for part one using the iris dataset with gradient descent.
2) multiclasslogisticregression.py is the code for part two using SGD on the MNIST data.
3) ALL COMMENTARY ON PART 2 IS BELOW!

  
The accuracy we get for part 2 with the MNIST data (as printed out when code runs) is .8993--which is great!

The plot of the objective function value versus epoch for stochastic gradient descent is plotted via the program..

The very last part of the code shows the images as examples of the 'difficult' hand written digits to classify.

For example, the first image shows is classified as a 7, but it's actually a 8. Honestly, I can see how they classified it as a 7 because the 'figure 8' of an 8 is pretty smushed and the computer probably can't recognize that.

The second image, the first image shows almost a figure-8 and classified it as a 4, but it's actually a 2. Honestly, I can't see any type of number in this one, but I can see how they got a 4--because there's a circle/closed part to the image.

The third image was shows a 'z' and was classified as a 7 but it was actually a 2. I can see how they classified this as a 7 because up until the bottom, it looks like a 7...but I write my 2s the same way instead of with a loop so I could tell this was actually a 2.


All of these examples had high confidence in its prediction, but the prediction is incorrect-- and we see this 'high confidence' from the list of probabilities in line 132.

    probabilities[difficultExamples[0:5]]
