function epsilon(episode,coldness, ep_min=0,ep_max=1){
    return ep_min + (ep_max - ep_min).toFixed(10)*(Math.exp(-coldness * episode));
}

function plotEpsilonDecay(n, eps) {
  episodes = ([...Array(eps).keys()].map(x => x++));
  decay = ([...Array(eps).keys()].map(x => epsilon(x,n)));
  var  data = {
  x: episodes,
  y: decay,
  mode: 'lines',
  type: 'scatter'
  };

  Plotly.newPlot('decayPlot', [data]);
}

function banditMeanDistribution(k) {
  return tf.randomNormal([1, k],0.0,1.0,'float32');
}

function plotKBandit(n){
  kMeans = banditMeanDistribution(n)
  kMeansData = kMeans.dataSync();

  banditData = Array();

  for(k of kMeansData) {
    banditData.push(tf.randomNormal([1, 1000],k,1.0,'float32'));
  }

  plotData = Array();

  for (row of banditData) {
    var bandit = {
      y: row.dataSync(),
      type: 'box',
      name: 'Bandit '+(plotData.length+1)
    };
    plotData.push(bandit)
  }

  Plotly.newPlot('banditPlot', plotData);
}

function getRandomSubarray(arr, size) {
    var shuffled = arr.slice(0), i = arr.length, temp, index;
    while (i--) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
    }
    return shuffled.slice(0, size);
}

function getRandomIntInclusive(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min; //The maximum is inclusive and the minimum is inclusive 
}

function exploreVsExploit(episodes, steps, decay) {
  exploreEps = Array();
  exploitEps = Array();
  avgReward = Array();
  kMeans = banditMeanDistribution(10);
  kMeansData = kMeans.dataSync();

  banditData = Array();

  for(k of kMeansData) {
    banditData.push(tf.randomNormal([1, 1000],k,1.0,'float32'));
  }

  for (ep of Array(episodes).keys()) {
    explore = 0;
    exploit = 0;
    reward = 0;
    
    for (t of Array(steps).keys()) {
      r = Math.random();
      eps = epsilon(ep, decay);

      if(r >= eps) {
        exploit += 1;
        aValues = Array(10).fill().map((_,i) => banditData[i].dataSync()[getRandomIntInclusive(0,banditData[i].size-1)]);
        reward += Math.max(...aValues);
      }
      else {
        explore += 1;
        action = getRandomIntInclusive(0,9);
        aValue = banditData[action].dataSync()[getRandomIntInclusive(0,banditData[action].size-1)];
        reward += aValue
      }
    }
    exploreEps.push(explore);
    exploitEps.push(exploit);
    avgReward.push(reward/steps)
  }
return [exploreEps, exploitEps, avgReward];
}

function plotExploreExploit(eps,steps,decay){

  results = exploreVsExploit(eps,steps,decay)
  var reward = {
      y: results[2],
      type: 'Scatter',
      name: 'Reward'
    };

  plotData = [reward]

  Plotly.newPlot('rewardPlot', plotData);

  var exploit = {
    y: results[1],
    type: 'Scatter',
    name: 'Exploitation'
  };

  var explore = {
    y: results[0],
    type: 'Scatter',
    name: 'Exploration'
  };

  plotData = [explore,exploit]
  Plotly.newPlot('evsePlot', plotData);

}

var app6 = new Vue({
  el: '#app',
  vuetify: new Vuetify(),
  data: {
    armnumber: '10',
    episodes: '500',
    steps: '50',
    epsDecay: '0.0009',
    decay: '0.1',
    showDecay: false,
    showBandit: false,
    showExploreExploit: false,
    drawer: false,
    mini: true
  },
  methods: {
    plotBandit: function () {
      plotKBandit(parseInt(this.armnumber));
    },
    plotDecay: function () {
      plotEpsilonDecay(parseFloat(this.epsDecay),10000);
    },
    plotExplore: function () {
      plotExploreExploit(parseInt(this.episodes),parseInt(this.steps),parseFloat(this.decay));
    }

  }
})
