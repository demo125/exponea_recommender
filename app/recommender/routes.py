from flask import jsonify, request, Blueprint, Flask

from recommender.controller import Controller

recommender_blueprint = Blueprint('recommender', __name__)


controller = Controller()

#train all models
@recommender_blueprint.route("/recommender/train", methods=['GET'])
def calculate():
    global controller
    controller.train()
    return 'trained'

#returns list of n most similar product to product_id
@recommender_blueprint.route("/similarity/<int:product_id>", methods=['GET'])
def get_n_most_similar_products(product_id):
    global controller
    response = controller.get_n_most_similar_products(product_id, __get_n_from_request(request))
    return jsonify(response)

#returns similarity between two products
@recommender_blueprint.route("/similarity/<int:product_id1>/<int:product_id2>", methods=['GET'])
def get_similarity_between_products(product_id1, product_id2):
    global controller
    response = controller.get_similarity_between_products(product_id1, product_id2)
    return jsonify(response)

#returns recommended products for customer_id
@recommender_blueprint.route("/recommender/<int:customer_id>", methods=['GET'])
def get_recommendations(customer_id, n=50):
    global controller

    response = controller.get_recommendations(customer_id, __get_n_from_request(request))
    return jsonify(response)


def __get_n_from_request(request):
    try:
        n = int(request.args.get('n'))
    except:
        n = 50
    return n




