const fs = require('fs')
const mnist = require('mnist')

const {test} = mnist.set(0, 1000)
const dir = 'public/data'
fs.mkdirSync(dir, {recursive: true})
test.forEach((e, i) => {
  fs.writeFileSync(`${dir}/sample-${i}.json`, JSON.stringify({
    pixels: e.input.map(n => Math.round(n * 255) / 255),
    label: e.output.indexOf(1),
  }))
})
