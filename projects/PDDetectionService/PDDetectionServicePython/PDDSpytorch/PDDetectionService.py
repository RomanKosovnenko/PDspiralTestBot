from flask_restful import Resource

class PDDetection(Resource):
    
  def get(self):
      return {
          "status" : "ok"
      }
  def post(self):
    
    return {
        "type" : "POST",
        "status" : "ok"
      }