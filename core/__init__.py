from core.utiles import *

class HelpingApi:
   
    def __init__(self):
        self._vectordb = FaceVectorStore()
        self._faceVector = FaceProcessor(face_model_path="face_dect/face-detection-0200.xml",
                        embed_model_path="face_emd/arcfaceresnet100-8.xml" , vector_db=FaceVectorStore())

if __name__ == '__main__':
    print('done')