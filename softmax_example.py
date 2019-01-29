import math

# What is really going on when I use softmax as an activation function?

def my_softmax(z):
    """ Just an example of how softmax works for my own edification
    Don't use this in any production code, stricky informational / educational.
    """
    # math.exp() = e**x where:
    # The value of e (Euler's Number - the base of the natural log) is raised to
    # the power of x (our numeric expression(s))  # http://bit.ly/exp_euler

    # Simple list comprehension applying math.exp over the input list
    z_exp = [math.exp(i) for i in z]

    # Add up all those numbers in z_exp
    sum_z_exp = sum(z_exp)

    # For All the items in the z_exp list devide them by sum_z_exp
    softmax = [i / sum_z_exp for i in z_exp]

    # return a list of rounded softmax values corresponding the the input list
    return ([round(i, 3) for i in softmax])


z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]

print(my_softmax(z))

# [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
