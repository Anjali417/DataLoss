import json
import boto3
import email
import uuid
import io

workmail = boto3.client('workmailmessageflow', region_name='eu-west-1')
runtime = boto3.client('runtime.sagemaker')
ENDPOINT_NAME = 'huggingface-pytorch-inference-2023-05-07-22-30-58-341'
s3 = boto3.client('s3')
textract = boto3.client('textract')


def lambda_handler(event, context):

    msg_id = event['messageId']
    raw_msg = workmail.get_raw_message_content(messageId=msg_id)

    parsed_msg = email.message_from_bytes(raw_msg['messageContent'].read())

    from_address = event['envelope']['mailFrom']['address']
    to_address = event['envelope']['recipients'][0]['address']

    # Extract the email body
    body = ''
    for part in parsed_msg.walk():
        if part.get_content_type() == 'text/plain':
            payload = part.get_payload(decode=True).decode('utf-8', 'ignore')
            body = payload.split('\n\n', 1)[-1]  # Remove email headers
            break

    domain_name = to_address.split('@')[1]

    texts = []

    texts.append(body)

    # Extract the attachments
    attachments = []

    for part in parsed_msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue
        filename = part.get_filename()
        if not filename:
            continue
        attachment = {
            'filename': filename,
            'data': part.get_payload(decode=True)
        }
        attachments.append(attachment)

    if domain_name != "u2264601.com":

        # Save the attachments to an S3 bucket

        bucket_name = 'u2264601'

        for attachment in attachments:

            s3_filename = str(uuid.uuid4()) + '-' + attachment['filename']
            s3.upload_fileobj(io.BytesIO(
                attachment['data']), bucket_name, s3_filename)

            print('Attachment: ', s3_filename)

            file_key = s3_filename

            # Call the Textract API to extract text from the PDF
            response = textract.detect_document_text(
                Document={'S3Object': {'Bucket': bucket_name, 'Name': file_key}})

            # Extract the text from the response
            text = response['Blocks']
            text = '\n'.join([t['Text']
                             for t in text if t['BlockType'] == 'LINE'])
            texts.append(text)

        print('Have Email Body & ', len(texts)-1, 'Attachments.')

        concatenated_text = "\n".join(texts)

        payload = concatenated_text

        data = {"inputs": payload, }

        my_json = json.dumps(data)

        # Convert the JSON string to a byte array
        my_bytes = my_json.encode('utf-8')

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME, ContentType='application/json', Body=my_bytes)

        result = json.loads(response['Body'].read().decode())

        # Count the number of elements in the "found" list
        pii_count = len(result['found'])

    if pii_count > 2:

        print('PII Count: ', pii_count)
        print('PII Detected; Email Bounced back.')

        return {
            'actions': [
                {
                    'action': {
                        'type': 'BOUNCE',
                        'parameters': {"bounceMessage": "Email in breach of company's GDPR policy."}
                    },
                    'allRecipients': True
                }
            ]}

    else:

        print('PII Not Detected; Email Sent to Recipient.')

        return {
            'actions': [
                {
                    'action': {
                        'type': 'DEFAULT',
                    },
                    'allRecipients': True
                }
            ]}
