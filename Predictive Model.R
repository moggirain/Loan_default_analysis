# Loaded the data with "Import Dataset" option and named the dataset 'default'
# Let's take a look at the overall structure of the data
str(default)

# Convert date (Variable issue_date is seen as a Factor by R because it only
# has a month and a year (e.g. Mar-14). To fix this, we paste "01" (the day) 
# in front of every date so that we can use the as.Date function.)
default$issue_date = paste("01", default$issue_date, sep="-")
default$issue_date = as.Date(default$issue_date, format="%d-%b-%y")

# Convert employee_title to characters so we can perform text mining on it
# and perform some descriptive analytics
default$employee_title = as.character(default$employee_title)
str(default$employee_title) # Now it's of the type "chr"

# Load the tm package and create the volatile corpus
library(tm)
employeetitles = VCorpus(VectorSource(default$employee_title))

# Let's take a look at all the employee titles
# This data is very dirty: some titles are all caps, some are not
# (e.g. there is "owner", "OWNER" and "Owner")
# some titles mean the same things but are spelled differently/incorrect
# (e.g. "nurse", "nure practitioner")
for (i in 1:length(employeetitles)) print(as.character(employeetitles[[i]]))

# Now let's do some transformations
# Everything to lower case
employeetitles = tm_map(employeetitles, content_transformer(tolower))
# Remove punctuation
employeetitles = tm_map(employeetitles, content_transformer(removePunctuation))
# Remove numbers
employeetitles = tm_map(employeetitles, content_transformer(removeNumbers))
# Remove stopwords
employeetitles = tm_map(employeetitles, removeWords, stopwords("english"))

# Stemming
library(SnowballC)
# Take a copy of the current employeetitles, since SnowballC is very destructive
employeetitles.dict = employeetitles
# Stem + remove white space
employeetitles = tm_map(employeetitles, stemDocument)
employeetitles = tm_map(employeetitles, content_transformer(stripWhitespace))

# Build a dataframe of all preprocessed employee titles
employeetitles.df = data.frame(text=unlist(sapply(employeetitles, `[`, "content")), stringsAsFactors=F)
# Replace the original employee titles with these ones
# This allows us to create a better predictive model
default$employee_title = employeetitles.df$text

# Now let's build a document term matrix
employeetitles.dtm = DocumentTermMatrix(employeetitles)
dim(employeetitles.dtm) # There are 1360 "documents" (e.g. employee titles) and 1147 terms
employeetitles.dtm

# Let's find the most frequent terms (e.g. with frequency > 20)
findFreqTerms(employeetitles.dtm, lowfreq=20)
# We can see from the data that there are a lot of managers, accountants,
# engineers, supervisors, teachers, etc. in this dataset.

# Now let's get the full details on terms
term.freq = colSums(as.matrix(employeetitles.dtm))
term.freq
term.freq[order(term.freq, decreasing=T)] # In order of decreasing frequency
# Let's make a dataframe of it (because ggplot only works with dataframes)
term.df = data.frame(word=names(term.freq), freq = term.freq)

# Let's plot the frequency of the terms - but only for terms with freq > 20
library(ggplot2)
ggplot(subset(term.df, freq>20), aes(word, freq)) + 
  geom_bar(stat="identity") +
  theme(axis.text.x=element_text(angle=45, hjust=1, size=20))
# We can see from the data that "manager" is the most frequent employee
# title in the dataset. Of course, this can be any kind of manager:
# IT manager, project manager, sales manager, etc.

# Now let's make a wordcloud!
library(wordcloud)
# Let's first complete the stemmed words so that they appear natural
term.df$cword = stemCompletion(term.df$word, dictionary=employeetitles.dict)
# Pick some colors
pal = c("#d40a0c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00","#ffff33", "#a65628")
# And create the wordcloud
wordcloud(term.df$cword, term.df$freq, min.freq=15, colors=pal, rot.per=0.2)


# Now for our predictive model, create a training & test set
# Let's first convert the employee_title variable back to a Factor type
default$employee_title = as.factor(default$employee_title)
library(rpart)
set.seed(563)
train = sample(1:nrow(default),nrow(default)*0.667)
default.train = default[train,]
default.test = default[-train,]

# Build the model
fit = rpart(default ~ annual_income + employee_length + grade + home_ownership + 
              loan_amount + nr_mortgages + months_since_last_delinquency + 
              nr_active_bank_accounts + nr_derogatory_public_records + purpose + 
              term_months + total_balance, # See explanation below
            data=default.train, method="class", # Train on the training set
            control=rpart.control(xval=10, minsplit=5), # 10-fold cross validation
            parms=list(split="information")) # Let's use the entropy

# Explanation for the variables used in our model:
# First of all, we left out the following variables: issue_date, remaining_principal
# and total_received_late_fees. The reasoning behind this is that, whenever we want
# to predict whether a new client is going to default on their loan or not, there
# is no data available for these variables (e.g. there is no issue date because
# the loan hasn't been issued yet, there is no remaining principal yet and no fees
# for the same reason as described above).
# Next, we left out higly correlated variables which we have found thanks to
# our visualizations in Tableau. There is a high correlation between "grade"
# and "interest_rate" (higher grade = higher interest rate), so we left out
# the latter. 
# The same goes for "monthly_installment" and "loan_amount". Obviously, 
# the higher the loan amount, the higher the monthly installment, so
# we again left out the latter variable.
# Another one is "nr_bankruptcies" and "nr_derogatory_public_records" since
# bankruptcies appear on your public record, so we left out the nr_bankruptcies
# variable.
# Lastly, we left out zip_code and employee_title. This is because these variables
# contain hundreds of unique values, which results in a huge importance assigned
# to these variables, which is somewhat misleading. One can test this by including
# these variables in the model anyway, which will result in a very high accuracy
# on the training set (about 99%) but a low accuracy on the test set. This is
# what's called overfitting and should be avoided at any cost.

# Create an accuracy function so we can easily calculate accuracy
accuracy = function(cm){
  return(sum(diag(cm))/sum(cm))
}

# TRAINING SET ACCURACY
# Extract the vectors of predicted and actual values for default
default.pred = predict(fit, default.train, type="class")
default.actual = default.train[,"default"]

# Build the confusion matrix
confusion.matrix = table(default.actual, default.pred)
confusion.matrix
addmargins(confusion.matrix)

# Accuracy
accuracytrain = accuracy(confusion.matrix)

# TEST SET ACCURACY
# Extract the vectors of predicted and actual values for default
default.pred = predict(fit, default.test, type="class")
default.actual = default.test[,"default"]

# Build the confusion matrix
confusion.matrix = table(default.actual, default.pred)
confusion.matrix
addmargins(confusion.matrix)

# Accuracy
accuracytest = accuracy(confusion.matrix)

# Show both accuracies in one matrix
accuracies = c(accuracytrain, accuracytest)
accuracy.matrix = matrix(accuracies, nrow=1)
colnames(accuracy.matrix) = c("Training Set", "Test Set")
rownames(accuracy.matrix) = "Accuracy"
accuracy.matrix
