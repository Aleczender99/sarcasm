from flask import Flask, render_template, request
import twitter
import process

app = Flask(__name__, template_folder="Pages", static_folder="Pages")


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/page1')
def page1():
    hashtag = request.args.get('hashtag')
    model = request.args.get('model')
    database = request.args.get('database')
    try:
        tweets = twitter.fetch_tweets(hashtag)
    except:
        tweets = twitter.fetch_local(hashtag)
    pred_tweets = process.predict(database, model, tweets)
    return render_template('page1.html', tweets=pred_tweets)


@app.route('/page2')
def page2():
    selected_tweet = request.args.get('selected_tweet')
    selected_tweet_id = request.args.get('selected_tweet_id')
    return render_template('page2.html', selected_tweet=selected_tweet, selected_tweet_id=selected_tweet_id)


@app.route('/submit_reply', methods=['GET'])
def submit_reply():
    reply = request.args.get('reply', '')
    tweet_id = request.args.get('selected_tweet_id', None)
    address = 'https://twitter.com/user/status/' + tweet_id
    print('Replying with: "', reply, '" to the tweet ', address)
    # twitter.respond_to_tweet(reply, tweet_id)

    return {'result': 'success'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
