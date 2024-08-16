import pymongo
from pymongo import MongoClient

class MongoDBManager:
    def __init__(self, connection_string, database_name):
        # Initialize MongoDB client
        self.client = MongoClient(connection_string)
        # Initialize database
        self.db = self.client[database_name]

    # Function to add data to MongoDB
    def add_data_to_mongodb(self, collection_name, document_id, data):
        try:
            collection = self.db[collection_name]
            data['_id'] = document_id  # MongoDB uses '_id' as the unique identifier for documents
            collection.insert_one(data)
            print(f'Data added to {collection_name}/{document_id}')
        except pymongo.errors.DuplicateKeyError:
            self.update_data_in_mongodb(collection_name, document_id, data)
            print(f'Document with id {document_id} already exists in {collection_name}')
        except Exception as e:
            print(f'Error adding data to MongoDB: {e}')

    def read_single_document(self, collection_name, document_id):
        try:
            collection = self.db[collection_name]
            doc = collection.find_one({'_id': document_id})
            if doc:
                print(f'Document data: {doc}')
            else:
                print(f'No such document: {collection_name}/{document_id}')
            return doc
        except Exception as e:
            print(f'Error reading document: {e}')
            return None
            
    def read_all_documents_in_collection(self, collection_name):
        try:
            collection = self.db[collection_name]
            docs = collection.find()
            all_docs = {doc['_id']: doc for doc in docs}
            return all_docs
        except Exception as e:
            print(f'Error reading documents: {e}')
            return None

    def update_data_in_mongodb(self, collection_name, document_id, data):
        try:
            collection = self.db[collection_name]
            old_data = self.read_single_document(collection_name, document_id)
            for key in data.keys():
                for key2 in data[key].keys():
                    if key in old_data.keys() and key2 in old_data[key].keys() and len(old_data[key][key2])== len(data[key][key2]):
                        print("nothing to update")
                        break
                    if key in old_data.keys() and key2 in old_data[key].keys():
                        old_data[key].extend(data[key])
                    else:
                        if key in old_data.keys():
                            old_data[key] = {}
                        old_data[key] = data[key]       
            collection.update_one({'_id': document_id}, {'$set': old_data})
            print(f'Data updated in {collection_name}/{document_id}')
        except Exception as e:
            print(f'Error updating data in MongoDB: {e}')