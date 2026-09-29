import axios from 'axios';
import {
  InvestigationReport,
  TestCase,
  ModelInfoResponse,
  GNNModelInfoResponse,
  EvidencePackage,
  Chain,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE || '';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export const api = {
  checkHealth: async (): Promise<{ status: string }> => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  getTestCases: async (): Promise<TestCase[]> => {
    const res = await apiClient.get<TestCase[]>('/api/v1/demo/test-cases');
    return res.data;
  },

  investigate: async (chain: Chain | string, address: string, maxHops: number = 6): Promise<InvestigationReport> => {
    const res = await apiClient.post<InvestigationReport>(
      `/api/v1/agentic/investigate/${chain}/${address}?max_hops=${maxHops}`
    );
    return res.data;
  },

  getEvidenceJson: async (chain: Chain | string, address: string): Promise<EvidencePackage> => {
    const res = await apiClient.get<EvidencePackage>(
      `/api/v1/evidence/${chain}/${address}?format=json`
    );
    return res.data;
  },

  downloadEvidenceMarkdown: async (chain: Chain | string, address: string): Promise<string> => {
    const res = await apiClient.get<string>(
      `/api/v1/evidence/${chain}/${address}?format=markdown`,
      { responseType: 'text' }
    );
    return res.data;
  },

  getXGBoostModelInfo: async (): Promise<ModelInfoResponse> => {
    const res = await apiClient.get<ModelInfoResponse>('/api/v1/ml/model-info');
    return res.data;
  },

  getGNNModelInfo: async (): Promise<GNNModelInfoResponse> => {
    const res = await apiClient.get<GNNModelInfoResponse>('/api/v1/gnn/model-info');
    return res.data;
  },
};
