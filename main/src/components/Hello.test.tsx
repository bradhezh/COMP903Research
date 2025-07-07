/// <reference types="vitest/globals" />

import {render, screen} from '@testing-library/react'
import '@testing-library/jest-dom'

import Hello from './Hello'

// `test` is global both in js and ts
test('Hello', () => {
  render(<Hello name="Brad" />)
  expect(screen.getByText('Hello, Brad!')).toBeInTheDocument()
})
