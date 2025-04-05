import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders learn react link', () => {
  render(<App />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
import '@testing-library/jest-dom';

function expect(linkElement: HTMLElement) {
  return {
    toBeInTheDocument: () => {
      if (!document.body.contains(linkElement)) {
        throw new Error('Expected element to be in the document');
      }
    }
  };
}
