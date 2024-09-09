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
                old_data= {key: value for key, value in loaded_old_data.items() if key != '_id'}
            
            print(old_data.keys())
            for profession, nested_data in data.items():
                if profession not in old_data:
                    old_data[profession] = {}
                for language, values in nested_data.items():
                    if language in old_data[profession]:
                        existing_values_set = set(map(json.dumps,old_data[profession][language]))
                        new_values_set = set(map(json.dumps,values))
                        old_data[profession][language]= list(map(json.loads,existing_values_set.union(new_values_set)))
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
                try:       
                    loaded_data = json.loads(file.read().decode('utf-8'))
                    all_docs[file.filename[:-5]] = {key: value for key, value in loaded_data.items() if key != '_id'}
                except Exception as doc_error:
                    print(f"Erreur lors de la lecture du document ID {file._id}: {doc_error}")
            return all_docs
        except Exception as e:
            print(f'Error reading documents: {e}')
            return None
    
    def find_and_solve_error(self):
        print("Find and solve error")
        try:
            for grid_out in self.fs.find():
                try:
                    data = grid_out.read()
                    # Traitez les données ici, par exemple les afficher
                    # print(f"Document ID: {grid_out._id}, Data: {data}")
                except Exception as doc_error:
                    print(f"Erreur lors de la lecture du document ID {grid_out._id}: {doc_error}")
        except Exception as e:
            print(f"Erreur lors de la lecture de la collection: {e}")
            
    
    """
The active selection is a Python class called MongoDBManager. This class is responsible for managing interactions with a MongoDB database. It provides methods for adding data to the database, reading documents from the database, and updating data in the database.

The __init__ method is the constructor of the class. It takes two parameters: connection_string and database_name. The connection_string is used to establish a connection to the MongoDB server, while the database_name specifies the name of the database to work with. Inside the constructor, the class initializes a MongoDB client using the MongoClient class from the pymongo library, and assigns it to the client attribute. It also initializes the db attribute by accessing the specified database using the database_name.

The add_data_to_mongodb method is used to add data to a MongoDB collection. It takes three parameters: collection_name, document_id, and data. The collection_name specifies the name of the collection to add data to, document_id is the unique identifier for the document, and data is a dictionary containing the data to be added. Inside the method, it first retrieves the collection from the database using the collection_name. It then adds the document_id to the data dictionary, as MongoDB uses the _id field as the unique identifier for documents. Finally, it inserts the data into the collection using the insert_one method. If a pymongo.errors.DuplicateKeyError is raised, it calls the update_data_in_mongodb method to update the existing document with the same document_id. Any other exceptions are caught and an error message is printed.

The read_single_document method is used to retrieve a single document from a specified collection in the database. It takes two parameters: collection_name and document_id. Inside the method, it retrieves the collection from the database using the collection_name and uses the find_one method to find the document with the specified document_id. If the document is found, it prints the document data. If the document is not found, it prints a message indicating that the document does not exist. The method returns the retrieved document as a dictionary or None if the document does not exist. Any exceptions are caught and an error message is printed.

The read_all_documents_in_collection method is used to retrieve all documents from a specified collection in the database. It takes one parameter: collection_name. Inside the method, it retrieves the collection from the database using the collection_name and uses the find method to retrieve all documents in the collection. It then creates a dictionary all_docs where the document ID is the key and the document itself is the value. The method returns this dictionary. If any exceptions occur, an error message is printed and None is returned.

The update_data_in_mongodb method is used to update data in a MongoDB collection. It takes three parameters: collection_name, document_id, and data. Inside the method, it retrieves the collection from the database using the collection_name and calls the read_single_document method to get the existing data for the specified document_id. It then iterates over the keys in the data dictionary and checks if the keys and their corresponding values in the existing data match the ones in the data dictionary. If they match, it prints a message indicating that there is nothing to update. If they don't match, it updates the existing data with the new data. Finally, it uses the update_one method to update the document in the collection. Any exceptions are caught and an error message is printed.

Overall, the MongoDBManager class provides a convenient interface for interacting with a MongoDB database, allowing users to add, read, and update data in a collection. It handles potential errors and provides informative messages for troubleshooting.


"""
    
    
    
    