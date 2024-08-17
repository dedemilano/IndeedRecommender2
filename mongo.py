import pymongo
from pymongo import MongoClient
import gridfs
import json

class MongoDBManager:
    def __init__(self, connection_string, database_name):
        # Initialize MongoDB client
        self.client = MongoClient(connection_string)
        # Initialize database
        self.db = self.client[database_name]

        self.fs = gridfs.GridFS(self.db)
        
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
            all_docs = {doc['_id']: {key: value for key, value in doc.items() if key != '_id'} for doc in docs}
            return all_docs
        except Exception as e:
            print(f'Error reading documents: {e}')
            return None

    def update_data_in_mongodb(self, collection_name, document_id, data):
        try:
            collection = self.db[collection_name]
            old_data = self.read_single_document(collection_name, document_id)

            for key, nested_data in data.items():
                if key not in old_data:
                    old_data[key] = {}
                for key2, values in nested_data.items():
                    if key2 in old_data[key] and len(old_data[key][key2]) == len(values):
                        print("Nothing to update")
                        continue
                    if key2 in old_data[key]:
                        old_data[key][key2].extend(values)
                    else:
                        old_data[key][key2] = values

            collection.update_one({'_id': document_id}, {'$set': old_data})
            print(f'Data updated in {collection_name}/{document_id}')

        except Exception as e:
            print(f'Error updating data in MongoDB: {e}')


    def add_data_like_json_with_GridFS(self, document_id, data):
        try:
            data['_id'] = document_id
            data = json.dumps(data)
            data = data.encode('utf-8')
            file_id = self.fs.put(data, filename=document_id+'.json')
            print(f'Data added to {document_id} with file_id: {file_id}')
        except Exception as e:
            print(f'Error adding data to MongoDB: {e}')
    
    def update_data_before_save_new_json_with_GridFS(self, document_id, data):
        try:
            old_data={}
            existing_file = self.fs.find_one({"filename": document_id+".json"})
            if existing_file:
                loaded_old_data = json.loads(existing_file.read().decode('utf-8'))
                old_data= {profession: value for profession, value in loaded_old_data.items() if profession != '_id'}
            
            print(old_data.keys())
            for profession, nested_data in data.items():
                if profession not in old_data:
                    old_data[profession] = {}
                for language, values in nested_data.items():
                    if language in old_data[profession] and len(old_data[profession][language]) == len(values):
                        print("Nothing to update")
                        continue
                    if language in old_data[profession]:
                        old_data[profession][language].extend(values)
                    else:
                        old_data[profession][language] = values
            
            if existing_file:
                self.fs.delete(existing_file._id)
            self.add_data_like_json_with_GridFS(document_id, old_data)
            print("")
        except Exception as e:
            print(f'Error updating data in MongoDB: {e}')
    
    def read_single_document_with_GridFS(self, document_id):
        try:
            existing_file = self.fs.find_one({"filename": document_id+".json"}).read()
            if existing_file:
                loaded_data = json.loads(existing_file.decode('utf-8'))
                print(f'Document data: {loaded_data}')
            else:
                print(f'No such document: {document_id}')
            return loaded_data
        except Exception as e:
            print(f'Error reading document: {e}')
            return None
        
    def read_all_documents_in_collection_with_GridFS(self):
        try:
            all_docs = {}
            for file in self.fs.find():
                loaded_data = json.loads(file.read().decode('utf-8'))
                all_docs[file.filename[:-5]] = {profession: value for profession, value in loaded_data.items() if profession != '_id'}
            return all_docs
        except Exception as e:
            print(f'Error reading documents: {e}')
            return None
    
    
    
    
    
    
    
    
    