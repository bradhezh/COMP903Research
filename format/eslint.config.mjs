import js from '@eslint/js'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default tseslint.config({ignores: ['dist']}, {
  extends: [js.configs.recommended, ...tseslint.configs.recommendedTypeChecked],
  files: ['**/*.ts'],
  languageOptions: {
    ecmaVersion: 2020,
    globals: globals.node,
    parserOptions: {
      project: true,
      tsconfigRootDir: import.meta.dirname,
    },
  },
  rules: {
    'no-trailing-spaces': 'warn',
    'linebreak-style': ['warn', 'unix'],
    'indent': ['warn', 2],
    'quotes': ['warn', 'single'],
    'semi': ['warn', 'never'],
    'eqeqeq': 'error',
    'no-constant-condition': 'warn',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-unsafe-member-access': 'warn',
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'off',
    '@typescript-eslint/restrict-template-expressions': 'off',
    '@typescript-eslint/restrict-plus-operands': 'off',
    '@typescript-eslint/no-unused-vars': ['error', {'argsIgnorePattern': '^_'}],
  },
})
