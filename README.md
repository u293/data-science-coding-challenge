# Uptain Data Science Coding Challenge

I will divide my proposed solution into several sections. Let's start with the first section.

# Analysis

We assume that these emails may contain information such as first name, last name, year of birth, and other information.
To extract such information, I will use regular expressions as they provide an efficient extraction method. However,
before we start with that, we need to discover the patterns in which these emails are written. i suggest that each email
address follows the following generic pattern:

One or more TEXTUAL tokens and/or one or more NUMERICAL tokens and/or one or more SEPARATOR characters @ DOMAIN

The TEXTUAL tokens have mostly the information about the name, first, last or middle name in some cases. Names can be
useful in many cases to know the age range or generation of a person, due to the fact that a certain name may be very
common in a certain time period and place. However, in order to make a wider use of such information, there should be an
additional external source of demographic information. In our case, we will consider extracting as much useful
information as possible from the textual tokens. the following is what I am thinking about and how it can be relevant to
our task:

- First Name
  As mentioned above, a name may be more common in a particular generation.
- Length of first name
  Older names tend to be longer. This can play a role.
- Number of name/text tokens
- the length of the username
- is the username uppercase (although it doesn't affect the functionality of the email, some official forms use
  uppercase)
- is the username separated by . or - or _.
  I assume that the method a user uses to choose his/her username may be affected by the formality and age maturity of
  the person. for example: a username like "dummyname1234" may be more common among the younger generations. Note: in
  this case, the year of email creation is important to guess the age of the person, it can also be included in the
  username. On the other hand, a username like "firstname.lastname" looks more official or serious, which indicates to a
  different age class. we will not go with the username semantics, of course this is useful in this task, but it
  requires more external sources.

I will only consider the first token of the names here, assuming it will be the first name of the person, I will encode
it to a numeric value. A possible relationship between the name and the generation will be shown later in the training
and evaluation of the models. Other names will be ignored, although they may play an indirect role for the task, but as
mentioned above, further analysis in this area requires more external sources.

The DOMAIN attribute can also be useful in this task, taking into account that a domain may be more popular for a
particular time or generation. Also, it may indicate a country or be more commonly used in one place than another,
therefore this will help to determine the person's country and consequently the possible generation of the name. for
this category I will consider:

- The mail provider
- The type of domain: for example, '.edu' indicates that a person is affiliated with an educational institution, which
  means that he/she is
  also belongs to a certain age group.

The NUMERICAL tokens are the most important part, because they can contain the person's birth year, which will be used
as basis for the TARGET attribute of the model. We assume that there are two common ways to write the birthdate year
here: either in 4 digits, e.g. 1999, or in 2 digits, e.g. 99. In both cases, we should only accept a reasonable range of
birth years, say 1924 - 1940. of birth years, say 1924 - 2006 (assuming a person's age is between 18 and 100). Any
number outside this range may have different or meaningless information. (Note that there can be different
interpretations of the numerical tokens, such as 2280 pointing to February 2, 1980, but I will consider these cases
ignorable and rare). In all cases we will extract:

- The number of numeric tokens.
- The numeric token that may indicate the year of birth.
- The length of the token that indicate to the age.

# Attributes Extraction

Before the extraction, I will filter out the invalid email addresses. Although these may contain valuable information,
processing the invalid addresses would also leave many missing or invalid attribute values that would require a fill or
fix technique. I decided to filter out the invalid addresses.

after removing the invalid address, i will use a regex to extract the features. with some processing on the numerical
values, we will decide if the person's age can be calculated, if yes, we assign the appropriate age category, if not, we
assign the category "unsure".

# Model training and evaluating

We will split the dataset into training and test sets in a cross-validation setup (5 folds) and hire different ML models
and test which one gives the best results. I will use traditional ML methods here because of the limited size of the
dataset. Deep learning methods usually require larger datasets.

# Running the code

For the env setup, I used python3.8 with pandas, sklearn and numpy. With "interactive_script.py" you can replicate the
experiment and test the model with new data.



