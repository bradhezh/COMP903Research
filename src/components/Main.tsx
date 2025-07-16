import * as tf from '@tensorflow/tfjs'
import * as ort from 'onnxruntime-web'
import {useState, useEffect, useRef} from 'react'

import {Label} from '@/components/ui/label'
import {Input} from '@/components/ui/input'
import {Button} from '@/components/ui/button'
import {Card, CardContent, CardHeader, CardTitle} from '@/components/ui/card'
import {ScrollArea} from '@/components/ui/scroll-area'

export const Main = () => {
  const [sampleCount, setSampleCount] = useState(100)
  const [running, setRunning] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const logRef = useRef<HTMLSpanElement | null>(null)

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

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollIntoView({behavior: "smooth"})
    }
  }, [logs])

  return (
    <div className="h-screen flex items-center justify-center px-4">
      <Card className="w-full max-w-xl">
        <CardHeader><CardTitle>Benchmark Tester</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="sampleCount">Number of test samples</Label>
            <Input id="sampleCount" type="number" value={sampleCount}
              onChange={e => setSampleCount(parseInt(e.target.value))} />
            <Button onClick={runBenchmark} disabled={running}>
              {running ? 'Running...' : 'Start Benchmark'}
            </Button>
          </div>
          <div>
            <h2 className="font-semibold text-sm mb-2">Log</h2>
            <ScrollArea className="h-80 rounded border bg-muted p-2">
              <pre className="text-sm whitespace-pre-wrap">
                {logs.join('\n')}<span ref={logRef}></span>
              </pre>
            </ScrollArea>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default Main
