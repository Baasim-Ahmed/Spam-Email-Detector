import pandas as pd

# Create synthetic spam and ham emails
spam_emails = [
    "Congratulations! You've won a free gift card. Click here to claim now.",
    "Urgent! Your account has been compromised. Please reset your password.",
    "Exclusive deal! Buy now and get 50% off on your next purchase.",
]

ham_emails = [
    "Hey, how are you? Let's catch up this weekend.",
    "Meeting scheduled at 10 AM tomorrow. Please confirm your availability.",
    "Thank you for your payment. Your receipt is attached.",
]

# Create the dataset
data = {
    'text': spam_emails + ham_emails,
    'label': ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails),
}

# Save as CSV
df = pd.DataFrame(data)
df.to_csv('emails.csv', index=False)
print("Dataset created and saved as emails.csv")
