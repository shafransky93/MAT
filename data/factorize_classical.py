# Function to factorize a large number
def factorize(n):
    # List to store the factors
    factors = []
    
    # Check every number from 2 to the square root of the number
    for i in range(2, int(n**0.5)+1):
        # If the remainder is 0, then the number is a factor
        if n % i == 0:
            factors.append(i)
            factors.append(n//i)
    
    # Return the list of factors
    return factors

# Take a large number as input
n = int(input("Enter a large number: "))

# Print the factors
print("The factors of {} are: {}".format(n, factorize(n)))
