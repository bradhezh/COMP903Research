import {useState} from 'react'
import * as tf from '@tensorflow/tfjs'
import * as ort from 'onnxruntime-web'

export const Main = () => {
  const [sampleCount, setSampleCount] = useState(100)
  const [running, setRunning] = useState(false)
  const [logs, setLogs] = useState<string[]>([])

  const log = (line: string) => {
    setLogs((prev) => [...prev, line])
  }

  const exportCSV = (data: any[]) => {
    const headers = Object.keys(data[0])
    const rows = data.map(row => headers.map(h => row[h]).join(','))
    const csv = [headers.join(','), ...rows].join('\n')
    const blob = new Blob([csv], {type: 'text/csv'})
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'results.csv'
    a.click()
  }

  const runBenchmark = async () => {
    setRunning(true)
    setLogs([])
    const results: any[] = []

    log('Test started.')
    const sample0 = await fetch(`/data/sample-0.json`).then(res => res.json())
    const pixelsTf0 = sample0.pixels
    const inputTf0 = tf.tensor(pixelsTf0).reshape([1, 28, 28, 1])
    const pixelsOrt0 = Float32Array.from(sample0.pixels)
    const inputOrt0 = new ort.Tensor('float32', pixelsOrt0, [1, 28, 28, 1])
    for (let i = 0; i < sampleCount; ++i) {
      log(`Sample ${i} ...`)
      // sample
      const sample =
        await fetch(`/data/sample-${i}.json`).then(res => res.json())
      const label = sample.label

      // tf
      const pixelsTf = sample.pixels
      const inputTf = tf.tensor(pixelsTf).reshape([1, 28, 28, 1])

      // load
      let t0 = performance.now()
      const modelTf = await tf.loadGraphModel(`/model/tfjs/model.json?ts=${t0}`)
      await (modelTf.predict(inputTf0) as tf.Tensor).argMax(-1).data()
      let t1 = performance.now()
      const loadTimeTf = (t1 - t0).toFixed(2)
      log(`Tf Loading time: ${loadTimeTf} ms`)

      // inference
      const predTf =
        await (modelTf.predict(inputTf) as tf.Tensor).argMax(-1).data()
      let t2 = performance.now()
      const inferTimeTf = (t2 - t1).toFixed(2)
      log(`Tf Inference time: ${inferTimeTf} ms`)
      const correctTf = predTf[0] === label
      log(`Tf Prediction: ${predTf[0]}, Actual: ${label}`)

      // ort
      const pixelsOrt = Float32Array.from(sample.pixels)
      const inputOrt = new ort.Tensor('float32', pixelsOrt, [1, 28, 28, 1])

      // load
      t0 = performance.now()
      const modelOrt =
        await ort.InferenceSession.create(`/model/onnx/mnist.onnx?ts=${t0}`)
      await modelOrt.run({keras_tensor: inputOrt0})
      t1 = performance.now()
      const loadTimeOrt = (t1 - t0).toFixed(2)
      log(`Ort Loading time: ${loadTimeOrt} ms`)

      // inference
      const outputOrt = await modelOrt.run({keras_tensor: inputOrt})
      const scoresOrt =
        outputOrt[Object.keys(outputOrt)[0]].data as Float32Array
      const predOrt = scoresOrt.indexOf(Math.max(...scoresOrt))
      t2 = performance.now()
      const inferTimeOrt = (t2 - t1).toFixed(2)
      log(`Ort Inference time: ${inferTimeOrt} ms`)
      const correctOrt = predOrt === label
      log(`Ort Prediction: ${predOrt}, Actual: ${label}`)

      results.push({
        i, label,
        loadTimeTf, loadTimeOrt, inferTimeTf, inferTimeOrt,
        correctTf, correctOrt,
      })
    }
    exportCSV(results)
    setRunning(false)
    log('Benchmark complete. CSV downloaded.')
  }

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-semibold">Benchmark Tester</h1>
      <div className="space-y-2">
        <label className="block">Number of test samples:
          <input className="border p-1 ml-2" type="number" value={sampleCount}
            onChange={e => setSampleCount(parseInt(e.target.value))} />
        </label>
        <button
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          onClick={runBenchmark} disabled={running} >
          {running ? 'Running...' : 'Start Benchmark'}
        </button>
      </div>
      <div className="mt-4">
        <h2 className="font-medium">Log</h2>
        <pre className="bg-gray-100 p-2 max-h-64 overflow-auto text-sm">
          {logs.join('\n')}
        </pre>
      </div>
    </div>
  )
}

export default Main
