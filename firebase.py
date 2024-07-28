import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

class FirebaseManager:
    def __init__(self, credential_path): 
        # Initialize Firebase Admin SDK
        if not firebase_admin._apps:
            cred = credentials.Certificate(credential_path)
            firebase_admin.initialize_app(cred)
        # Initialize Firestore DB
        self.db = firestore.client()

    # Function to add data to Firestore
    def add_data_to_firestore(self, collection_name, document_id, data):
        try:
            doc_ref = self.db.collection(collection_name).document(document_id)
            doc_ref.set(data)
            print(f'Data added to {collection_name}/{document_id}')
        except Exception as e:
            print(f'Error adding data to Firestore: {e}')

    def read_single_document(self, collection_name, document_id):
        try:
            doc_ref = self.db.collection(collection_name).document(document_id)
            doc = doc_ref.get()
            if doc.exists:
                print(f'Document data: {doc.to_dict()}')
            else:
                print(f'No such document: {collection_name}/{document_id}')
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            print(f'Error reading document: {e}')
            return None
            
    def read_all_documents_in_collection(self, collection_name):
        try:
            docs = self.db.collection(collection_name).stream()
            all_docs = {doc.id: doc.to_dict() for doc in docs}
            return all_docs
        except Exception as e:
            print(f'Error reading documents: {e}')
            return None
