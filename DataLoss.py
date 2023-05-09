from faker import Faker
import random
import os
import csv
import spacy
import re
import email
import nltk
import pandas as pd
from spacy.training.example import Example
import warnings

# Ignore all user warnings globally
warnings.simplefilter("ignore", UserWarning)

fake = Faker('en_GB')  # Use the 'en_GB' locale for UK-specific data

email_files = []


def FindFilePaths(path):

    for root, dirs, files in os.walk(path):
        for file in files:
            if "." not in file:
                email_files.append(os.path.join(root, file))


def ProcessFiles(email_file):

    with open(email_file, 'r') as f:
        raw_email = f.read()
        email_message = email.message_from_string(raw_email)
        body = ""
        for part in email_message.walk():
            if part.get_content_type() == 'text/plain':
                body += part.get_payload()
    return body


def GenNewEmail():

    # Generate a random number of emails to use for the paragraph
    num_emails = random.randint(5, 10)
    email_bodies = []
    for i in range(num_emails):
        email_file = random.choice(email_files)
        email_body = ProcessFiles(email_file)
        email_bodies.append(email_body)

    # Concatenate the email bodies to form a single paragraph
    paragraph = "\n\n".join(email_bodies)

    # Tokenize the paragraph into sentences using NLTK
    sentences = nltk.sent_tokenize(paragraph)

    # Generate a random number of sentences to use for the final paragraph
    num_sentences = random.randint(3, 5)

    # Select random sentences from the original paragraph to form the final paragraph
    final_sentences = []
    for i in range(num_sentences):
        sentence = random.choice(sentences)
        final_sentences.append(sentence)

    # Concatenate the final sentences to form the final paragraph
    final_paragraph = " ".join(final_sentences)

    # define the regular expression pattern to match URLs
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    # remove the URLs from the text paragraph
    final_paragraph = re.sub(url_pattern, '', final_paragraph)

    return final_paragraph


def GenName():

    # Randomly choose whether to generate a single name or full name
    rand = fake.boolean()

    if rand:
        # Generate a name
        name = fake.name()
    else:
        if rand:
            name = fake.first_name()
        else:
            name = fake.last_name()

    return name


def GenEmail():

    # Generate an email with a random domain name
    email = fake.email(domain=fake.domain_name())
    return email


def GenPhone():
    # Randomly choose to generate a phone number in different formats
    rand = fake.boolean()

    phone_number = ''

    if rand:
        mobile_prefix = '07'
        mobile_suffix = ''.join([str(random.randint(0, 9)) for i in range(8)])
        mobile_number = f'{mobile_prefix}{mobile_suffix}'

        # Randomly select a format for the mobile number
        mobile_format = random.choice(['07{} {}'.format(mobile_suffix[:3], mobile_suffix[3:]),
                                       '+44 7{} {}'.format(
                                           mobile_suffix[:3], mobile_suffix[3:]),
                                       '07{}-{}'.format(mobile_suffix[:3], mobile_suffix[3:])])
        phone_number = mobile_format

    else:
        # List of 10 UK area codes
        area_codes = ['01', '02', '03', '05', '07',
                      '08', '0161', '01642', '01752', '020']
        # Generate a random landline number within the selected area code
        area_code = random.choice(area_codes)
        landline_number = f'{area_code}{random.randint(10000, 999999)}'
        # Randomly select a format for the landline number
        landline_format = random.choice(['{} {}'.format(landline_number[:5], landline_number[5:]),
                                         '{}-{}'.format(
                                             landline_number[:5], landline_number[5:]),
                                         '+44 {} {}'.format(
                                             landline_number[:4], landline_number[4:]),
                                         '+44-{}-{}'.format(landline_number[:4], landline_number[4:])])
        phone_number = landline_format

    return str(phone_number)


def GenNHS():

    # Generate a random NHS number
    nhs_number = fake.numerify('##########')

    # Choose a random format
    formats = ['nnn nnn nnnn', 'nnnnnnnnnn', 'nnn-nnn-nnnn', 'nnn.nnn.nnnn']
    format_choice = random.choice(formats)

    # Format the NHS number accordingly
    if format_choice == 'nnn nnn nnnn':
        formatted_number = '{} {} {}'.format(
            nhs_number[:3], nhs_number[3:6], nhs_number[6:])
    elif format_choice == 'nnnnnnnnnn':
        formatted_number = nhs_number
    elif format_choice == 'nnn-nnn-nnnn':
        formatted_number = '{}-{}-{}'.format(
            nhs_number[:3], nhs_number[3:6], nhs_number[6:])
    else:
        formatted_number = '{}.{}.{}'.format(
            nhs_number[:3], nhs_number[3:6], nhs_number[6:])

    return str(formatted_number)


def GenAddress():

    # generate a random Building Number
    house_number = fake.building_number()
    # randomly choose either 'A' or 'a' ' '
    letter = random.choice(['A', 'a', ''])

    house_number = str(house_number)+letter

    uk_address = house_number + " " + fake.street_name() + " " + \
        fake.city() + " " + fake.postcode()

    return uk_address


def GenCard():

    # Choose a random format
    formats = ['XXXX XXXX XXXX XXXX',
               'XXXX-XXXX-XXXX-XXXX', 'XXXXXXXXXXXXXXXX']
    chosen_format = random.choice(formats)

    # Choose a random format
    cards = ['visa', 'mastercard', 'amex']
    card_choice = random.choice(cards)
    # Generate a credit card number
    credit_card_number = fake.credit_card_number(card_type=card_choice)

    # Format the credit card number according to the chosen format
    if chosen_format == 'XXXX XXXX XXXX XXXX':
        formatted_card_number = ' '.join(
            [credit_card_number[i:i+4] for i in range(0, 16, 4)])
    elif chosen_format == 'XXXX-XXXX-XXXX-XXXX':
        formatted_card_number = '-'.join([credit_card_number[i:i+4]
                                         for i in range(0, 16, 4)])
    else:
        formatted_card_number = credit_card_number.replace(' ', '')

    return str(formatted_card_number)


def GenBank():

    bank_account_number = fake.random_number(digits=8)

    return str(bank_account_number)


def GenSortCode():

    sort_code = str(fake.random_number(digits=6))

    # randomly choose one of the two formats
    format_choice = random.choice(["xxxxxx", "xx-xx-xx"])

    # format the sort code accordingly
    if format_choice == "xx-xx-xx":
        sort_code = '{}-{}-{}'.format(
            sort_code[:2], sort_code[2:4], sort_code[4:6])

    return str(sort_code)


def GenDOB():

    # Generate a random date of birth and format it as '01 Jan 1990'
    dob = fake.date_of_birth().strftime('%d %b %Y')

    return str(dob)


def GenKeywords():

    KeyWords = []
    # Generate a random number between 0 and 9
    random_number = random.randint(0, 10)

    # Generate a random number between 1 and 2
    random_number_2 = random.randint(1, 2)

    # Create an if-elseif statement based on the generated random number
    if random_number == 0:

        # Create a for loop based on the generated random number
        for i in range(random_number_2):
            KeyWords.append(['SortCode', GenSortCode()])

    elif random_number == 1:

        # Create a for loop based on the generated random number
        for i in range(random_number_2):
            KeyWords.append(['BankAccount', GenBank()])
            KeyWords.append(['SortCode', GenSortCode()])

    elif random_number == 2:

        # Create a for loop based on the generated random number
        for i in range(random_number_2):
            KeyWords.append(['NHS', GenNHS()])
            KeyWords.append(['BankAccount', GenBank()])
            KeyWords.append(['SortCode', GenSortCode()])

    elif random_number == 3:

        # Create a for loop based on the generated random number
        for i in range(random_number_2):
            KeyWords.append(['CreditCard', GenCard()])
            KeyWords.append(['NHS', GenNHS()])
            KeyWords.append(['BankAccount', GenBank()])
            KeyWords.append(['SortCode', GenSortCode()])

    elif random_number == 4:

        # Create a for loop based on the generated random number
        for i in range(random_number_2):
            KeyWords.append(['PhoneNo', GenPhone()])
            KeyWords.append(['CreditCard', GenCard()])
            KeyWords.append(['NHS', GenNHS()])
            KeyWords.append(['BankAccount', GenBank()])
            KeyWords.append(['SortCode', GenSortCode()])

    elif random_number == 5:

        # Create a for loop based on the generated random number
        for i in range(random_number_2):
            KeyWords.append(['Address', GenAddress()])
            KeyWords.append(['PhoneNo', GenPhone()])
            KeyWords.append(['CreditCard', GenCard()])
            KeyWords.append(['NHS', GenNHS()])
            KeyWords.append(['BankAccount', GenBank()])
            KeyWords.append(['SortCode', GenSortCode()])

    elif random_number == 6:

        # Create a for loop based on the generated random number
        for i in range(random_number_2):
            KeyWords.append(['Name', GenName()])
            KeyWords.append(['Address', GenAddress()])
            KeyWords.append(['PhoneNo', GenPhone()])
            KeyWords.append(['CreditCard', GenCard()])
            KeyWords.append(['NHS', GenNHS()])
            KeyWords.append(['BankAccount', GenBank()])
            KeyWords.append(['SortCode', GenSortCode()])

    elif random_number == 7:

        # Create a for loop based on the generated random number
        for i in range(random_number_2):
            KeyWords.append(['Email', GenEmail()])
            KeyWords.append(['Name', GenName()])
            KeyWords.append(['Address', GenAddress()])
            KeyWords.append(['PhoneNo', GenPhone()])
            KeyWords.append(['CreditCard', GenCard()])
            KeyWords.append(['NHS', GenNHS()])
            KeyWords.append(['BankAccount', GenBank()])
            KeyWords.append(['SortCode', GenSortCode()])

    elif random_number == 8:

        # Create a for loop based on the generated random number
        for i in range(random_number_2):
            KeyWords.append(['DOB', GenDOB()])
            KeyWords.append(['Email', GenEmail()])
            KeyWords.append(['Name', GenName()])
            KeyWords.append(['Address', GenAddress()])
            KeyWords.append(['PhoneNo', GenPhone()])
            KeyWords.append(['CreditCard', GenCard()])
            KeyWords.append(['NHS', GenNHS()])
            KeyWords.append(['BankAccount', GenBank()])
            KeyWords.append(['SortCode', GenSortCode()])

    else:

        KeyWords.append([' ', ' '])

    return KeyWords


def GetIndices(keywords, paragraph):

    used_indices = set()

    for keyword in keywords:
        start_index = paragraph.find(keyword[1])
        if start_index != -1:
            end_index = start_index + len(keyword[1]) - 1

            used_indices.add((start_index, end_index))

    return used_indices


def IntroduceNewKeywords():

    KeyWords = GenKeywords()
    KeyWords2 = KeyWords.copy()
    final_paragraph = GenNewEmail()

# Insert keywords into random non-overlapping places in the paragraph
    used_indices = set()

    # Generate a random index that does not overlap with any previous keyword
    index = random.randint(0, len(final_paragraph) - 1)

    final_paragraph = final_paragraph[:index] + ' ' + \
        KeyWords[0][1] + ' ' + final_paragraph[index+len(KeyWords[0][1]):]

    used_indices = GetIndices(KeyWords2, final_paragraph)

    del KeyWords[0]

    for keyword in KeyWords:

        length = len(keyword[1])

        # Generate a random index that does not overlap with any previous keyword
        start = random.randint(0, len(final_paragraph) - 1)

        end = start + length

        for start, end in used_indices:
            start = random.randint(0, len(final_paragraph) - 1)
            end = start + length

        final_paragraph = final_paragraph[:start] + ' ' + \
            keyword[1] + ' ' + final_paragraph[start+len(keyword[1]):]

        used_indices = GetIndices(KeyWords2, final_paragraph)

    return FindIndices(KeyWords2, final_paragraph)


def FindIndices(keywords, paragraph):

    keyword_list = []

    for keyword in keywords:
        start_index = paragraph.find(keyword[1])
        if start_index != -1:
            end_index = start_index + len(keyword[1]) - 1

            item = {'Label': keyword[0], 'Keyword': keyword[1],
                    'Start': start_index, 'End': end_index}

            keyword_list.append(item)

    # append a new row
    return pd.Series({'Text': paragraph, 'Features': keyword_list})


FindFilePaths('maildir')

# create empty dataframe
df = pd.DataFrame(columns=["Text", "Features"])

for i in range(100):

    pd_series = IntroduceNewKeywords()
    delete_row = False
    # Keep track of seen start and end values
    seen_start = set()
    seen_end = set()

    for feat in pd_series['Features']:
        start = feat['Start']
        end = feat['End']
        # Check if start or end value has been seen before
        if start in seen_start or end in seen_end:
            delete_row = True
            break
        seen_start.add(start)
        seen_end.add(end)

    if not delete_row:
        # append the new row to the DataFrame
        df = pd.concat([df, pd_series.to_frame().T],
                       ignore_index=True)
    else:
        print('Delete')

    i = i+1

    # Clear seen sets for next row
    seen_start.clear()
    seen_end.clear()


# Load the spaCy model
nlp = spacy.blank('en')

# Define the new labels
LABELS = ['Name', 'Email', 'NHS', 'DOB', 'Address',
          'PhoneNo', 'CreditCard', 'BankAccount', 'SortCode']

# Add the new labels to the model
ner = nlp.create_pipe("ner")
for label in LABELS:
    ner.add_label(label)
nlp.add_pipe('ner', last=True)
nlp.replace_pipe("ner", "ner")
nlp.initialize()

# Load the training data from the Pandas DataFrame
train_data = []
for i, row in df.iterrows():
    text = row['Text']
    features = row['Features']

    ents = []

    for feat in features:
        label = feat['Label']
        start = feat['Start']
        end = feat['End']
        keyword = feat['Keyword']
        ents.append((start, end, label))

    doc = nlp(text)

    spans = []

    for start, end, label in ents:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            spans.append(span)
    doc.spans['entities'] = spans
    train_data.append(Example.from_dict(doc, {'entities': spans}))


# Train the model


nlp.begin_training()

for i in range(20):
    losses = {}
    nlp.update(train_data, losses=losses, drop=0.2)
    print(losses)



score = nlp.evaluate(train_data)
# print(score)

# Save the trained model to disk
output_dir = "Save/"
nlp.to_disk(output_dir)
