from flask import jsonify, request, Blueprint, Flask

from recommender.controller import Controller

# from flask import current_app

recommender_blueprint = Blueprint('recommender', __name__)

controller = Controller()

@recommender_blueprint.route("/recommender/train", methods=['GET'])
def calculate():
    global controller
    controller.train()
    return 'trained'

@recommender_blueprint.route("/similarity/<int:product_id>", methods=['GET'])
def get_n_most_similar_products(product_id):
    global controller
    response = controller.get_n_most_similar_products(product_id, 50)
    return jsonify(response)

@recommender_blueprint.route("/similarity/<int:product_id1>/<int:product_id2>", methods=['GET'])
def get_similarity_between_products(product_id1, product_id2):
    global controller
    response = controller.get_similarity_between_products(product_id1, product_id2)
    return jsonify(response)

@recommender_blueprint.route("/recommender/<int:customer_id>", methods=['GET'])
def get_recommendations(customer_id, n=50):
    global controller
    response = controller.get_recommendations(customer_id, n)
    return jsonify(response)



