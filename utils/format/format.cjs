const fs = require('fs')
const mnist = require('mnist')

const {test} = mnist.set(0, 100)

test.forEach((e, i) => {
  fs.writeFileSync(`public/data/sample-${i}.json`, JSON.stringify({
    pixels: e.input.map(n => Math.round(n * 255) / 255),
    label: e.output.indexOf(1),
  }))
})
