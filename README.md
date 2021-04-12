# MATH373_Spring2021_Project2_ammuth

*Note*
  The last chunk of code in the multiclasslogisticregression.py file should only be run seperately after the rest of the code runs. 
  
The accuracy we get for part 2 with the MNIST data (as printed out when code runs) is .9227--which is great!

The plot of the objective function value versus epoch for stochastic gradient descent is plotted via the program..

The very last part of the code shows the images as examples of the 'difficult' hand written digits to classify.

For example, the first image shows almost a figure-8 and classified it as a 4, but it's actually a 2. Honestly, I can't see any type of number in this one, but I can see how they got a 4--because there's a circle/closed part to the image.

The second image was shows a 'z' and was classified as a 7 but it was actually a 2. I can see how they classified this as a 7 because up until the bottom, it looks like a 7...but I write my 2s the same way instead of with a loop so I could tell this was actually a 2.

The third image shows a 7 with a line thorugh it and was classified as a 2 but it was a 7. I also write my 7s the same way as this image, so I could tell what number it actually was, but I can see how they thought it was a 2 based on the last image shown--the 2s written almost like a 'z'.

All of these examples had high confidence in its prediction, but the prediction is incorrect-- and we see this 'high confidence' from the list of probabilities in line 132.

    probabilities[difficultExamples[0:5]]
